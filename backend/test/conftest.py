"""Shared fixtures and environment setup for the backend test suite.

Three things belong here rather than in a test module:

* **Settings before the app is imported.** `get_settings` is `lru_cache`d and
  `src.main` resolves settings while it is being imported, so the environment has
  to be configured *first*. A `conftest` module is imported before the test
  modules beside it, which is the one place where "first" is guaranteed. The API
  tests used to each set this up themselves, which is also why importing them in
  the wrong order silently worked against the wrong temp directory.
* **Scratch space.** Every test gets a directory of its own under one session
  root, so a test that counts session directories or writes an archive cannot see
  another test's files.
* **The shared builders.** A JPEG, a `.project` archive and a two-second test
  clip are each wanted by several suites, and each used to carry its own copy.

`conftest` is not imported by name from a test module (that is not what it is
for), so the builders are exposed as fixtures that *return a callable*.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]

#: One scratch root for the whole run; removed by `pytest_sessionfinish`.
SESSION_ROOT = Path(tempfile.mkdtemp(prefix="vsr-pytest-"))

# ── the environment, before any test module imports the app ─────────────────

# `setdefault` throughout: an operator can point the suite at a real directory,
# or a different interpreter's ffmpeg, without editing this file.
os.environ.setdefault("VSR_TEMP_DIR", str(SESSION_ROOT / "tmp"))
os.environ.setdefault("VSR_LOG_DIR", str(SESSION_ROOT / "logs"))
os.environ.setdefault("VSR_PROJECTS_DIR", str(SESSION_ROOT / "projects"))

# SAM 3 has to look *enabled*, because `/api/sam3/*` checks that before anything
# else and the routes under test are stubbed rather than skipped. The stub
# service answers every prompt and the real models stay on disk: the lifespan
# preloads them on the way up, so that preload is patched out in the
# `_stub_model_preload` fixture below.
os.environ.setdefault("SAM3_ENABLED", "1")

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

# `VSR_*` is what the app reads; the bare names are what the standalone scripts
# in `LEGACY_SCRIPTS` read. Both are set while the migration is in progress.
if FFMPEG:
    os.environ.setdefault("FFMPEG_BIN", FFMPEG)
    os.environ.setdefault("VSR_FFMPEG_BIN", FFMPEG)
if FFPROBE:
    os.environ.setdefault("FFPROBE_BIN", FFPROBE)
    os.environ.setdefault("VSR_FFPROBE_BIN", FFPROBE)


def pytest_sessionfinish(session, exitstatus) -> None:
    """Drop the scratch root, unless a run asked to keep it for inspection."""
    if os.environ.get("VSR_KEEP_SCRATCH"):
        print(f"\nscratch kept at {SESSION_ROOT}")
        return
    shutil.rmtree(SESSION_ROOT, ignore_errors=True)


# ── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session", autouse=True)
def _stub_model_preload():
    """Keep the lifespan from loading real SAM 3 weights during the suite.

    The app preloads its models unconditionally on startup, and the API tests
    enter that lifespan through `TestClient`. The stub service answers the
    requests instead, so the preload is replaced with a no-op here and the
    checkpoints are left on disk.
    """
    from unittest import mock

    with mock.patch("src.core.lifespan.warmup_models", return_value=[]):
        yield


@pytest.fixture(scope="session")
def backend_root() -> Path:
    """The backend directory, which is also what `src.*` resolves against."""
    return BACKEND_ROOT


@pytest.fixture
def scratch() -> Path:
    """A directory of this test's own, removed afterwards."""
    path = Path(tempfile.mkdtemp(prefix="case-", dir=SESSION_ROOT))
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def client_id() -> str:
    """The client id the API tests act as. See `core.identity`."""
    return "a" * 32


@pytest.fixture
def jpeg():
    """`jpeg(width, height, colour)` -> JPEG bytes."""

    def make(width=64, height=48, colour=(120, 40, 200)) -> bytes:
        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (width, height), colour).save(buffer, format="JPEG")
        return buffer.getvalue()

    return make


@pytest.fixture
def archive():
    """`archive(frames=, video=, dataset=, record=)` -> `.project` bytes.

    Built by hand rather than through `projects.builder`, so the reader is tested
    against the layout contract instead of against the writer. `record=False`
    leaves the annotation out entirely; the default synthesises the one a writer
    would have produced from the frame names.
    """

    def make(*, frames=None, video=None, dataset=None, record=True) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as bundle:
            for name, data in (frames or {}).items():
                bundle.writestr(f"frames/{name}", data)
            if video is not None:
                if isinstance(video, (bytes, bytearray)):
                    bundle.writestr("video/clip.mp4", video)
                else:
                    bundle.write(video, "video/clip.mp4")
            if dataset is None and record:
                dataset = {
                    "videos": [
                        {
                            "id": 1,
                            "video_name": "clip",
                            "file_names": list(frames or {}),
                            "fps": 6,
                            "segmentation_mode": "instance",
                        }
                    ],
                    "annotations": [],
                    "categories": [],
                }
            if dataset is not None:
                bundle.writestr("annotation.json", json.dumps(dataset))
        return buffer.getvalue()

    return make


@pytest.fixture(scope="session")
def ffmpeg() -> tuple[str, str]:
    """The ffmpeg and ffprobe paths, skipping the test when they are missing.

    A skip rather than a failure: extraction is a real dependency of the suite,
    but a machine without it should still be able to run everything else.
    """
    if not (FFMPEG and FFPROBE):
        pytest.skip("ffmpeg and ffprobe are required for this test")
    return FFMPEG, FFPROBE


@pytest.fixture(scope="session")
def video(ffmpeg) -> Path:
    """A two-second 320x240 clip at 25 fps: 50 source frames, 2.0 seconds.

    Built once for the session, since it costs a second of ffmpeg and several
    suites measure the same file.
    """
    path = SESSION_ROOT / "clip.mp4"
    if not path.is_file():
        subprocess.run(
            [
                ffmpeg[0],
                "-hide_banner",
                "-nostdin",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=320x240:rate=25:duration=2",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            check=True,
        )
    return path


@pytest.fixture
def api_client(client_id):
    """A `TestClient` for the app, already carrying the client-id header.

    Entered as a context manager so the app's lifespan runs — that is what
    exercises the start-up session sweep. The app is imported inside the fixture
    so a test that never speaks HTTP does not import it at all.
    """
    from fastapi.testclient import TestClient
    from src.main import app

    with TestClient(app) as client:
        client.headers["X-Vsr-Client"] = client_id
        yield client
