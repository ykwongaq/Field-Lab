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

from src.domain.windows import CHAININGS

DEFAULT_CONFIG_FILE = "config/server.json"

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_LOG_DIR = "logs"
DEFAULT_LOG_LEVEL = "INFO"

# Scratch space for in-flight uploads, and the directory that keeps the finished
# `.project` archives.
DEFAULT_TEMP_DIR = "tmp"
DEFAULT_PROJECTS_DIR = "projects"

# Frame rate a new project falls back to when the caller does not ask for one.
# Read by the batch builder and the CLI (`projects.builder.create_project`), not
# by `Settings`: the interactive path is the browser, which picks its own rate.
DEFAULT_TARGET_FPS = 6.0

DEFAULT_MAX_UPLOAD_BYTES = 4 * 1024**3  # 4 GiB project archive

# Frames one SAM 3 session may hold while propagating, how many of them are
# shared with the next window (the re-anchoring stretch), and how many of those
# shared frames are written into tracker memory. Peak GPU memory follows the
# window size, not the clip length, which is what lets a reviewer propagate a
# mask to the end of a long clip.
DEFAULT_PROPAGATE_WINDOW_FRAMES = 48
DEFAULT_PROPAGATE_OVERLAP = 8
DEFAULT_PROPAGATE_ANCHOR_MAX = 3
DEFAULT_PROPAGATE_CHAINING = "derived"
DEFAULT_PROPAGATE_MAX_JOBS = 32
#: How many of those jobs one client may hold at once.
#:
#: The global ceiling alone lets a single reviewer fill the whole queue and leave
#: everyone else unable to start a run. That is a fairness problem rather than a
#: capacity one: the GPU runs one job at a time either way, and every queued job
#: is holding its masks in host memory. Clamped to `max_jobs` by the registry, so
#: a small deployment cannot end up with an allowance larger than the queue.
DEFAULT_PROPAGATE_MAX_JOBS_PER_CLIENT = 8
DEFAULT_PROPAGATE_JOB_TTL_SECONDS = 1800

#: Run inference under bf16 autocast on CUDA (see `Sam3Config.autocast_enabled`).
DEFAULT_SAM3_AUTOCAST = True
#: Models kept resident at once. 2 holds both the image model and the video
#: predictor, which is what makes an eager start-up worth doing: the reviewer can
#: click and propagate without either load evicting the other. Drop it to 1 on a
#: small GPU, where the two will take turns instead.
DEFAULT_SAM3_MAX_RESIDENT_MODELS = 2
#: Frames whose vision embeddings stay cached for click-by-click prompting.
DEFAULT_SAM3_IMAGE_CACHE_SIZE = 2
#: How many reviewers keep such a cache. The budget is per client rather than
#: shared, because a single flat LRU lets one reviewer's clicks evict another's
#: and every eviction costs a full vision-backbone pass; the client count is what
#: keeps that fair *and* bounded. Start here: 1 reproduces the old single-cache
#: behaviour, and raising it multiplies peak host memory by the same factor.
DEFAULT_SAM3_IMAGE_CACHE_CLIENTS = 4

# A session is a disposable frame working copy under `temp_dir/sessions/<uuid>/`.
# It is a cache of something the archive can always regenerate, so an idle one
# may be swept at any time; the byte cap bounds how much disk all of them hold.
DEFAULT_SESSION_TTL_SECONDS = 6 * 60 * 60  # 6 hours idle
DEFAULT_SESSION_MAX_BYTES = 64 * 1024**3  # 64 GiB across every session
#: One client's share of that disk: an owner over this loses its own oldest
#: sessions and never another owner's, so one reviewer's large upload cannot
#: evict a colleague's open clip. Four reviewers fit inside the global cap.
DEFAULT_SESSION_MAX_BYTES_PER_CLIENT = 16 * 1024**3  # 16 GiB per client
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


def _file_bool(
    section: Mapping[str, Any], key: str, default: bool, *, label: str
) -> bool:
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

    # Logging: `log_level` applies to the `vsr` logger; `log_dir` is where its
    # rotating file handler writes. An empty `log_dir` keeps logging on stderr.
    log_dir: str = DEFAULT_LOG_DIR
    log_level: str = DEFAULT_LOG_LEVEL

    # Storage (both relative paths resolve against the process working directory)
    temp_dir: str = DEFAULT_TEMP_DIR
    projects_dir: str = DEFAULT_PROJECTS_DIR

    # HTTP limits
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    cors_origins: Tuple[str, ...] = ("*",)

    # SAM 3
    # `enable_sam3=False` keeps the process free of torch and sam3: no model is
    # probed or loaded and the SAM 3 endpoints answer 503.
    enable_sam3: bool = True
    #: Path to a local `sam3.pt`. `None` lets the package fetch the gated
    #: `facebook/sam3` checkpoint from the Hugging Face cache.
    sam3_checkpoint: Optional[str] = None
    #: Tokenizer vocabulary for a local checkpoint (the package asset otherwise).
    sam3_bpe_path: Optional[str] = None
    sam3_device: str = "auto"
    #: How many of the two models may be loaded at once (LRU).
    sam3_max_resident_models: int = DEFAULT_SAM3_MAX_RESIDENT_MODELS
    #: Instance interactivity is what makes point and box prompts possible.
    sam3_inst_interactivity: bool = True
    #: Wrap every model call in one bf16 autocast context (CUDA only).
    sam3_autocast: bool = DEFAULT_SAM3_AUTOCAST
    sam3_image_cache_size: int = DEFAULT_SAM3_IMAGE_CACHE_SIZE
    sam3_image_cache_clients: int = DEFAULT_SAM3_IMAGE_CACHE_CLIENTS
    sam3_eager: bool = False

    # Propagation
    propagate_window_frames: int = DEFAULT_PROPAGATE_WINDOW_FRAMES
    propagate_overlap: int = DEFAULT_PROPAGATE_OVERLAP
    propagate_anchor_max: int = DEFAULT_PROPAGATE_ANCHOR_MAX
    #: `derived` | `verified`: whether a window with no verified mask of its own
    #: may be seeded from the previous window's output. See `domain.windows`.
    propagate_chaining: str = DEFAULT_PROPAGATE_CHAINING
    propagate_max_jobs: int = DEFAULT_PROPAGATE_MAX_JOBS
    #: How many of those one client may hold (see the constant for why).
    propagate_max_jobs_per_client: int = DEFAULT_PROPAGATE_MAX_JOBS_PER_CLIENT
    propagate_job_ttl_seconds: int = DEFAULT_PROPAGATE_JOB_TTL_SECONDS

    # Sessions: the frame working copy every reader shares
    session_ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS
    session_max_bytes: int = DEFAULT_SESSION_MAX_BYTES
    session_max_bytes_per_client: int = DEFAULT_SESSION_MAX_BYTES_PER_CLIENT
    frames_jpeg_quality: int = DEFAULT_FRAMES_JPEG_QUALITY
    ffmpeg_bin: str = DEFAULT_FFMPEG_BIN
    ffprobe_bin: str = DEFAULT_FFPROBE_BIN

    def __post_init__(self) -> None:
        """Reject settings that could never work, before the service starts.

        A propagation window chain needs a positive stride, and the stride is
        `window_frames - overlap`, so an overlap that reaches the window size
        would leave the planner with nowhere to step. The image cache and the job
        ceiling are likewise meaningless at zero.

        The session and frame settings are checked here for a sharper reason than
        "it would not work": a TTL or a byte cap of zero would make the sweeper
        delete the very frames a reviewer is looking at, and a JPEG quality
        outside 1..100 would only fail later, inside ffmpeg's `-q:v` mapping,
        long after the value was read. Failing here means a bad
        `config/server.json` stops the service at startup instead of surfacing as
        a confusing error on the first request that happens to need the value.
        """
        if self.propagate_window_frames < 2:
            raise ValueError(
                "propagate.window_frames must be at least 2, got "
                f"{self.propagate_window_frames}."
            )
        if not 0 <= self.propagate_overlap < self.propagate_window_frames:
            raise ValueError(
                "propagate.overlap must be at least 0 and smaller than "
                f"propagate.window_frames ({self.propagate_window_frames}), got "
                f"{self.propagate_overlap}."
            )
        if self.propagate_anchor_max < 0:
            raise ValueError(
                f"propagate.anchor_max must be >= 0, got {self.propagate_anchor_max}."
            )
        if self.propagate_chaining not in CHAININGS:
            raise ValueError(
                f"propagate.chaining must be one of {', '.join(CHAININGS)}, got "
                f"{self.propagate_chaining!r}."
            )
        if self.propagate_max_jobs < 1:
            raise ValueError(
                f"propagate.max_jobs must be at least 1, got {self.propagate_max_jobs}."
            )
        if self.propagate_max_jobs_per_client < 1:
            raise ValueError(
                "propagate.max_jobs_per_client must be at least 1, got "
                f"{self.propagate_max_jobs_per_client}."
            )
        if self.sam3_max_resident_models < 1:
            raise ValueError(
                "sam3.max_resident_models must be at least 1, got "
                f"{self.sam3_max_resident_models}."
            )
        if self.sam3_image_cache_size < 1:
            raise ValueError(
                "sam3.image_cache_size must be at least 1, got "
                f"{self.sam3_image_cache_size}."
            )
        if self.sam3_image_cache_clients < 1:
            raise ValueError(
                "sam3.image_cache_clients must be at least 1, got "
                f"{self.sam3_image_cache_clients}."
            )
        if not 1 <= self.frames_jpeg_quality <= 100:
            raise ValueError(
                "frames_jpeg_quality must be within 1..100, got "
                f"{self.frames_jpeg_quality}."
            )
        if self.session_ttl_seconds < 1:
            raise ValueError(
                "session_ttl_seconds must be at least 1, got "
                f"{self.session_ttl_seconds}; zero would make every session "
                "sweepable the moment it is created."
            )
        if self.session_max_bytes < 1:
            raise ValueError(
                "session_max_bytes must be at least 1, got "
                f"{self.session_max_bytes}; zero would make the sweeper delete "
                "sessions to get back under the cap."
            )
        if self.session_max_bytes_per_client < 1:
            raise ValueError(
                "session_max_bytes_per_client must be at least 1, got "
                f"{self.session_max_bytes_per_client}; zero would make the "
                "sweeper delete every session its owner has."
            )
        if self.max_upload_bytes < 0:
            raise ValueError(
                "max_upload_bytes must be 0 (no limit) or greater, got "
                f"{self.max_upload_bytes}."
            )


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
            _file_str(
                logging_section, "log_dir", DEFAULT_LOG_DIR, label="logging.log_dir"
            ),
        ),
        log_level=_str_env(
            "VSR_LOG_LEVEL",
            _file_str(
                logging_section,
                "log_level",
                DEFAULT_LOG_LEVEL,
                label="logging.log_level",
            ),
        ),
        temp_dir=_str_env(
            "VSR_TEMP_DIR",
            _file_str(config, "temp_dir", DEFAULT_TEMP_DIR, label="temp_dir"),
        ),
        projects_dir=_str_env(
            "VSR_PROJECTS_DIR",
            _file_str(
                config, "projects_dir", DEFAULT_PROJECTS_DIR, label="projects_dir"
            ),
        ),
        max_upload_bytes=_int_env(
            "VSR_MAX_UPLOAD_BYTES",
            _file_int(
                http,
                "max_upload_bytes",
                DEFAULT_MAX_UPLOAD_BYTES,
                label="http.max_upload_bytes",
            ),
        ),
        cors_origins=_list_env(
            "VSR_CORS_ORIGINS",
            _file_str_list(server, "cors_origins", ("*",), label="server.cors_origins"),
        ),
        sam3_checkpoint=_str_env(
            "SAM3_CHECKPOINT",
            _file_str(sam3, "checkpoint", "", label="sam3.checkpoint"),
        )
        or None,
        enable_sam3=_bool_env(
            "SAM3_ENABLED", _file_bool(sam3, "enabled", True, label="sam3.enabled")
        ),
        sam3_device=_str_env(
            "SAM3_DEVICE", _file_str(sam3, "device", "auto", label="sam3.device")
        ),
        sam3_max_resident_models=_int_env(
            "SAM3_MAX_RESIDENT_MODELS",
            _file_int(
                sam3,
                "max_resident_models",
                DEFAULT_SAM3_MAX_RESIDENT_MODELS,
                label="sam3.max_resident_models",
            ),
        ),
        sam3_inst_interactivity=_bool_env(
            "SAM3_INST_INTERACTIVITY",
            _file_bool(
                sam3,
                "inst_interactivity",
                True,
                label="sam3.inst_interactivity",
            ),
        ),
        sam3_autocast=_bool_env(
            "SAM3_AUTOCAST",
            _file_bool(sam3, "autocast", DEFAULT_SAM3_AUTOCAST, label="sam3.autocast"),
        ),
        sam3_image_cache_size=_int_env(
            "SAM3_IMAGE_CACHE_SIZE",
            _file_int(
                sam3,
                "image_cache_size",
                DEFAULT_SAM3_IMAGE_CACHE_SIZE,
                label="sam3.image_cache_size",
            ),
        ),
        sam3_image_cache_clients=_int_env(
            "SAM3_IMAGE_CACHE_CLIENTS",
            _file_int(
                sam3,
                "image_cache_clients",
                DEFAULT_SAM3_IMAGE_CACHE_CLIENTS,
                label="sam3.image_cache_clients",
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
        propagate_window_frames=_int_env(
            "PROPAGATE_WINDOW_FRAMES",
            _file_int(
                propagate,
                "window_frames",
                DEFAULT_PROPAGATE_WINDOW_FRAMES,
                label="propagate.window_frames",
            ),
        ),
        propagate_overlap=_int_env(
            "PROPAGATE_OVERLAP",
            _file_int(
                propagate,
                "overlap",
                DEFAULT_PROPAGATE_OVERLAP,
                label="propagate.overlap",
            ),
        ),
        propagate_anchor_max=_int_env(
            "PROPAGATE_ANCHOR_MAX",
            _file_int(
                propagate,
                "anchor_max",
                DEFAULT_PROPAGATE_ANCHOR_MAX,
                label="propagate.anchor_max",
            ),
        ),
        propagate_chaining=_str_env(
            "PROPAGATE_CHAINING",
            _file_str(
                propagate,
                "chaining",
                DEFAULT_PROPAGATE_CHAINING,
                label="propagate.chaining",
            ),
        ),
        propagate_max_jobs=_int_env(
            "PROPAGATE_MAX_JOBS",
            _file_int(
                propagate,
                "max_jobs",
                DEFAULT_PROPAGATE_MAX_JOBS,
                label="propagate.max_jobs",
            ),
        ),
        propagate_max_jobs_per_client=_int_env(
            "PROPAGATE_MAX_JOBS_PER_CLIENT",
            _file_int(
                propagate,
                "max_jobs_per_client",
                DEFAULT_PROPAGATE_MAX_JOBS_PER_CLIENT,
                label="propagate.max_jobs_per_client",
            ),
        ),
        propagate_job_ttl_seconds=_int_env(
            "PROPAGATE_JOB_TTL_SECONDS",
            _file_int(
                propagate,
                "job_ttl_seconds",
                DEFAULT_PROPAGATE_JOB_TTL_SECONDS,
                label="propagate.job_ttl_seconds",
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
        session_max_bytes_per_client=_int_env(
            "VSR_SESSION_MAX_BYTES_PER_CLIENT",
            _file_int(
                sessions,
                "max_bytes_per_client",
                DEFAULT_SESSION_MAX_BYTES_PER_CLIENT,
                label="sessions.max_bytes_per_client",
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
            _file_str(config, "ffmpeg_bin", DEFAULT_FFMPEG_BIN, label="ffmpeg_bin"),
        ),
        ffprobe_bin=_str_env(
            "VSR_FFPROBE_BIN",
            _file_str(config, "ffprobe_bin", DEFAULT_FFPROBE_BIN, label="ffprobe_bin"),
        ),
    )
