"""Application settings.

Every tunable value is resolved here, exactly once, with this precedence:

1. an environment variable (e.g. ``VSR_MAX_UPLOAD_BYTES``),
2. ``config/server.json`` (see ``backend/config/server.json``),
3. the built-in defaults below.

The JSON file is taken from ``VSR_CONFIG_FILE`` when that is set, otherwise
``config/server.json`` next to the backend package. When the file is merely
absent the built-in defaults apply; when it is unreadable or malformed, loading
raises, so a typo cannot silently fall back to defaults. Nothing else in the
codebase should call `os.environ`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

DEFAULT_CONFIG_FILE = "config/server.json"

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_LOG_DIR = "logs"
DEFAULT_LOG_LEVEL = "INFO"

# Scratch space for in-flight uploads, and the directory that keeps the finished
# `.project` archives.
DEFAULT_TEMP_DIR = "tmp"
DEFAULT_PROJECTS_DIR = "projects"

# Frame rate assumed when a new project does not ask for one.
DEFAULT_TARGET_FPS = 6.0

DEFAULT_MAX_UPLOAD_BYTES = 4 * 1024**3  # 4 GiB source video
DEFAULT_MAX_FRAME_BYTES = 32 * 1024**2  # 32 MiB single frame

DEFAULT_SAM3_MODEL = "facebook/sam3"
DEFAULT_SAM3_THRESHOLD = 0.5
DEFAULT_SAM3_MASK_THRESHOLD = 0.5
DEFAULT_SAM3_EXEMPLAR_FRACTION = 0.06
DEFAULT_PROPAGATE_MAX_FRAMES = 120

# A session is a disposable frame working copy under `temp_dir/sessions/<uuid>/`.
# It is a cache of something the archive can always regenerate, so an idle one
# may be swept at any time; the byte cap bounds how much disk all of them hold.
DEFAULT_SESSION_TTL_SECONDS = 6 * 60 * 60  # 6 hours idle
DEFAULT_SESSION_MAX_BYTES = 64 * 1024**3  # 64 GiB across every session
DEFAULT_FRAMES_JPEG_QUALITY = 95
DEFAULT_FFMPEG_BIN = "ffmpeg"
DEFAULT_FFPROBE_BIN = "ffprobe"

_TRUTHY = {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in _TRUTHY


def _list_env(name: str, default: Tuple[str, ...]) -> Tuple[str, ...]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    items = tuple(part.strip() for part in raw.split(",") if part.strip())
    return items or default


def _str_env(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


def _config_path() -> Optional[Path]:
    """Locate the JSON config file, or `None` when there is none.

    `VSR_CONFIG_FILE` wins when set; otherwise `config/server.json` is looked up
    relative to the current directory and to the backend root, so the service
    behaves the same whether uvicorn starts from `backend/` or the repo root.
    """
    override = os.environ.get("VSR_CONFIG_FILE", "").strip()
    if override:
        return Path(override).expanduser()

    candidates = (
        Path.cwd() / DEFAULT_CONFIG_FILE,
        Path(__file__).resolve().parents[2] / DEFAULT_CONFIG_FILE,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _read_config_file() -> Mapping[str, Any]:
    """Read the JSON config file into a mapping (empty when there is none).

    Returns `{}` only when no config file is found by the automatic lookup. An
    unreadable or invalid file raises, so a typo cannot silently fall back to
    the built-in defaults.
    """
    path = _config_path()
    if path is None:
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Could not read the config file {path}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Config file {path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a JSON object.")
    return data


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Return one object of the config file (empty when the section is absent)."""
    value = config.get(name)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section `{name}` must be a JSON object.")
    return value


def _file_int(section: Mapping[str, Any], key: str, default: int, *, label: str) -> int:
    value = section.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer, got {value!r}.")
    return value


def _file_float(
    section: Mapping[str, Any], key: str, default: float, *, label: str
) -> float:
    value = section.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number, got {value!r}.")
    return float(value)


def _file_bool(section: Mapping[str, Any], key: str, default: bool, *, label: str) -> bool:
    value = section.get(key)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be true or false, got {value!r}.")
    return value


def _file_str(section: Mapping[str, Any], key: str, default: str, *, label: str) -> str:
    value = section.get(key)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string, got {value!r}.")
    return value


def _file_str_list(
    section: Mapping[str, Any], key: str, default: Tuple[str, ...], *, label: str
) -> Tuple[str, ...]:
    value = section.get(key)
    if value is None:
        return default
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a list of strings, got {value!r}.")
    return tuple(value) or default


@dataclass(frozen=True)
class Settings:
    """Resolved runtime configuration."""

    # Server (host/port are used by whatever launches uvicorn)
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT

    # Logging
    log_dir: str = DEFAULT_LOG_DIR
    log_level: str = DEFAULT_LOG_LEVEL

    # Storage (both relative paths resolve against the process working directory)
    temp_dir: str = DEFAULT_TEMP_DIR
    projects_dir: str = DEFAULT_PROJECTS_DIR

    # HTTP limits
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES
    cors_origins: Tuple[str, ...] = ("*",)

    # New projects: frame rate used when the caller does not choose one
    default_fps: float = DEFAULT_TARGET_FPS

    # SAM 3 (image concept segmentation)
    # `enable_sam3=False` keeps the process free of transformers/torch: no model
    # is probed or loaded and the SAM 3 endpoints answer 503.
    enable_sam3: bool = True
    sam3_model: str = DEFAULT_SAM3_MODEL
    sam3_device: str = "auto"
    sam3_dtype: str = "auto"
    sam3_threshold: float = DEFAULT_SAM3_THRESHOLD
    sam3_mask_threshold: float = DEFAULT_SAM3_MASK_THRESHOLD
    sam3_exemplar_fraction: float = DEFAULT_SAM3_EXEMPLAR_FRACTION
    sam3_eager: bool = False
    # Tokenizer vocabulary used when SAM 3 is loaded from a local checkpoint.
    sam3_bpe_path: Optional[str] = None

    # SAM 3 video tracker (mask propagation)
    sam3_tracker_model: Optional[str] = None
    propagate_max_frames: int = DEFAULT_PROPAGATE_MAX_FRAMES

    # Sessions: the frame working copy every reader shares
    session_ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS
    session_max_bytes: int = DEFAULT_SESSION_MAX_BYTES
    frames_jpeg_quality: int = DEFAULT_FRAMES_JPEG_QUALITY
    ffmpeg_bin: str = DEFAULT_FFMPEG_BIN
    ffprobe_bin: str = DEFAULT_FFPROBE_BIN


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build `Settings` (cached; call with care in tests).

    Precedence per value: environment variable > `config/server.json` > built-in
    default. The file read happens here, inside the cached function, so the JSON
    is parsed once per process.
    """
    config = _read_config_file()
    server = _section(config, "server")
    logging_section = _section(config, "logging")
    http = _section(config, "http")
    sam3 = _section(config, "sam3")
    video = _section(config, "video")
    propagate = _section(config, "propagate")
    sessions = _section(config, "sessions")

    return Settings(
        host=_str_env(
            "VSR_HOST", _file_str(server, "host", DEFAULT_HOST, label="server.host")
        ),
        port=_int_env(
            "VSR_PORT", _file_int(server, "port", DEFAULT_PORT, label="server.port")
        ),
        log_dir=_str_env(
            "VSR_LOG_DIR",
            _file_str(logging_section, "log_dir", DEFAULT_LOG_DIR, label="logging.log_dir"),
        ),
        log_level=_str_env(
            "VSR_LOG_LEVEL",
            _file_str(
                logging_section, "log_level", DEFAULT_LOG_LEVEL, label="logging.log_level"
            ),
        ),
        temp_dir=_str_env(
            "VSR_TEMP_DIR",
            _file_str(config, "temp_dir", DEFAULT_TEMP_DIR, label="temp_dir"),
        ),
        projects_dir=_str_env(
            "VSR_PROJECTS_DIR",
            _file_str(config, "projects_dir", DEFAULT_PROJECTS_DIR, label="projects_dir"),
        ),
        max_upload_bytes=_int_env(
            "VSR_MAX_UPLOAD_BYTES",
            _file_int(
                http, "max_upload_bytes", DEFAULT_MAX_UPLOAD_BYTES, label="http.max_upload_bytes"
            ),
        ),
        max_frame_bytes=_int_env(
            "VSR_MAX_FRAME_BYTES",
            _file_int(
                http, "max_frame_bytes", DEFAULT_MAX_FRAME_BYTES, label="http.max_frame_bytes"
            ),
        ),
        cors_origins=_list_env(
            "VSR_CORS_ORIGINS",
            _file_str_list(
                server, "cors_origins", ("*",), label="server.cors_origins"
            ),
        ),
        default_fps=_float_env(
            "VSR_DEFAULT_FPS",
            _file_float(video, "default_fps", DEFAULT_TARGET_FPS, label="video.default_fps"),
        ),
        sam3_model=_str_env(
            "SAM3_MODEL",
            _file_str(sam3, "model_path", DEFAULT_SAM3_MODEL, label="sam3.model_path"),
        ),
        enable_sam3=_bool_env(
            "SAM3_ENABLED", _file_bool(sam3, "enabled", True, label="sam3.enabled")
        ),
        sam3_device=_str_env(
            "SAM3_DEVICE", _file_str(sam3, "device", "auto", label="sam3.device")
        ),
        sam3_dtype=_str_env(
            "SAM3_DTYPE", _file_str(sam3, "dtype", "auto", label="sam3.dtype")
        ),
        sam3_threshold=_float_env(
            "SAM3_THRESHOLD",
            _file_float(
                sam3, "threshold", DEFAULT_SAM3_THRESHOLD, label="sam3.threshold"
            ),
        ),
        sam3_mask_threshold=_float_env(
            "SAM3_MASK_THRESHOLD",
            _file_float(
                sam3,
                "mask_threshold",
                DEFAULT_SAM3_MASK_THRESHOLD,
                label="sam3.mask_threshold",
            ),
        ),
        sam3_exemplar_fraction=_float_env(
            "SAM3_EXEMPLAR_BOX",
            _file_float(
                sam3,
                "exemplar_fraction",
                DEFAULT_SAM3_EXEMPLAR_FRACTION,
                label="sam3.exemplar_fraction",
            ),
        ),
        sam3_eager=_bool_env(
            "SAM3_EAGER", _file_bool(sam3, "eager", False, label="sam3.eager")
        ),
        sam3_bpe_path=_str_env(
            "SAM3_BPE_PATH",
            _file_str(sam3, "bpe_path", "", label="sam3.bpe_path"),
        )
        or None,
        sam3_tracker_model=_str_env(
            "SAM3_TRACKER_MODEL",
            _file_str(
                sam3, "tracker_model_path", "", label="sam3.tracker_model_path"
            ),
        )
        or None,
        propagate_max_frames=_int_env(
            "PROPAGATE_MAX_FRAMES",
            _file_int(
                propagate,
                "max_frames",
                DEFAULT_PROPAGATE_MAX_FRAMES,
                label="propagate.max_frames",
            ),
        ),
        session_ttl_seconds=_int_env(
            "VSR_SESSION_TTL_SECONDS",
            _file_int(
                sessions,
                "ttl_seconds",
                DEFAULT_SESSION_TTL_SECONDS,
                label="sessions.ttl_seconds",
            ),
        ),
        session_max_bytes=_int_env(
            "VSR_SESSION_MAX_BYTES",
            _file_int(
                sessions,
                "max_bytes",
                DEFAULT_SESSION_MAX_BYTES,
                label="sessions.max_bytes",
            ),
        ),
        frames_jpeg_quality=_int_env(
            "VSR_FRAMES_JPEG_QUALITY",
            _file_int(
                config,
                "frames_jpeg_quality",
                DEFAULT_FRAMES_JPEG_QUALITY,
                label="frames_jpeg_quality",
            ),
        ),
        ffmpeg_bin=_str_env(
            "VSR_FFMPEG_BIN",
            _file_str(
                config, "ffmpeg_bin", DEFAULT_FFMPEG_BIN, label="ffmpeg_bin"
            ),
        ),
        ffprobe_bin=_str_env(
            "VSR_FFPROBE_BIN",
            _file_str(
                config, "ffprobe_bin", DEFAULT_FFPROBE_BIN, label="ffprobe_bin"
            ),
        ),
    )
