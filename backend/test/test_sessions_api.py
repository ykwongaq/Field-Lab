"""HTTP checks for `/api/sessions`, using FastAPI's own test client.

Run from the backend root::

    python test/test_sessions_api.py

Needs ffmpeg/ffprobe on PATH, or ``FFMPEG_BIN`` / ``FFPROBE_BIN``. Settings are
forced to a throwaway temp dir before the app is imported, because
``get_settings`` is cached and ``main`` builds the app at import time.
"""

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_ROOT)

FFMPEG = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg")
FFPROBE = os.environ.get("FFPROBE_BIN") or shutil.which("ffprobe")
assert FFMPEG and os.path.isfile(FFMPEG), f"ffmpeg not found: {FFMPEG!r}"
assert FFPROBE and os.path.isfile(FFPROBE), f"ffprobe not found: {FFPROBE!r}"

# Configure before importing the app: `get_settings` is lru_cached and `main`
# resolves settings while it is being imported.
ROOT = tempfile.mkdtemp(prefix="vsr-api-")
TEMP_DIR = os.path.join(ROOT, "tmp")
os.environ["VSR_TEMP_DIR"] = TEMP_DIR
os.environ["VSR_LOG_DIR"] = os.path.join(ROOT, "logs")
os.environ["VSR_PROJECTS_DIR"] = os.path.join(ROOT, "projects")
os.environ["VSR_FFMPEG_BIN"] = FFMPEG
os.environ["VSR_FFPROBE_BIN"] = FFPROBE
os.environ["SAM3_ENABLED"] = "0"

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from src.main import app  # noqa: E402

checks = 0
SESSIONS_DIR = os.path.join(TEMP_DIR, "sessions")


def ok(label):
    global checks
    checks += 1
    print("  ok:", label)


def jpeg(width=48, height=32, colour=(90, 30, 160)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buffer, format="JPEG")
    return buffer.getvalue()


def frames_archive(frames, dataset=None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in frames.items():
            archive.writestr(f"frames/{name}", data)
        archive.writestr(
            "annotation.json",
            json.dumps(
                dataset
                if dataset is not None
                else {
                    "videos": [
                        {
                            "id": 1,
                            "video_name": "clip",
                            "file_names": list(frames),
                            "fps": 6,
                            "segmentation_mode": "instance",
                        }
                    ],
                    "annotations": [],
                    "categories": [],
                }
            ),
        )
    return buffer.getvalue()


def video_archive(video_path, dataset=None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.write(video_path, "video/clip.mp4")
        archive.writestr(
            "annotation.json",
            json.dumps(
                dataset
                if dataset is not None
                else {
                    "videos": [
                        {"video_name": "clip", "fps": 6, "target_fps": 6}
                    ]
                }
            ),
        )
    return buffer.getvalue()


def post(client, name, data):
    return client.post(
        "/api/sessions",
        files={"project": (name, data, "application/zip")},
    )


def session_dirs():
    if not os.path.isdir(SESSIONS_DIR):
        return []
    return sorted(os.listdir(SESSIONS_DIR))


def test_frames_round_trip(client):
    """A frame folder goes in, the same bytes come back out, in order."""
    frames = {
        "b.jpg": jpeg(colour=(10, 200, 10)),
        "a.jpg": jpeg(colour=(200, 10, 10)),
        "c.jpg": jpeg(colour=(10, 10, 200)),
    }
    response = post(client, "clip_007.project", frames_archive(frames))
    assert response.status_code == 201, response.text
    body = response.json()

    assert body["source"] == "frames", body["source"]
    assert body["frame_names"] == ["b.jpg", "a.jpg", "c.jpg"], body["frame_names"]
    assert body["frame_count"] == 3
    assert body["fps"] == 6, body["fps"]
    assert (body["width"], body["height"]) == (48, 32)
    assert body["mode"] == "instance"
    assert body["archive"] == "clip_007.project"
    assert body["recorded_frame_names"] == ["b.jpg", "a.jpg", "c.jpg"]
    assert body["frame_names_match"] is True
    ok("POST /api/sessions copies a frame folder and reports the sequence")

    session_id = body["session_id"]
    for index, name in enumerate(["b.jpg", "a.jpg", "c.jpg"]):
        frame = client.get(f"/api/sessions/{session_id}/frames/{index}")
        assert frame.status_code == 200, frame.text
        assert frame.content == frames[name], f"frame {index} bytes differ"
        assert frame.headers["content-type"] == "image/jpeg"
        assert "immutable" in frame.headers["cache-control"]
    ok("GET frames/{index} returns the stored bytes with immutable caching")

    again = client.get(f"/api/sessions/{session_id}")
    assert again.status_code == 200
    assert again.json()["frame_names"] == ["b.jpg", "a.jpg", "c.jpg"]
    ok("GET /api/sessions/{id} re-describes the session after a reload")

    missing = client.get(f"/api/sessions/{session_id}/frames/99")
    assert missing.status_code == 404, missing.status_code
    negative = client.get(f"/api/sessions/{session_id}/frames/-1")
    assert negative.status_code == 422, negative.status_code
    ok("an out-of-range frame is 404 and a negative index is 422")

    assert client.delete(f"/api/sessions/{session_id}").status_code == 204
    assert session_id not in session_dirs()
    assert client.delete(f"/api/sessions/{session_id}").status_code == 204
    ok("DELETE drops the session and is idempotent")


def test_video_round_trip(client, video_path):
    response = post(
        client,
        "from_video.project",
        video_archive(
            video_path,
            {
                "videos": [
                    {
                        "video_name": "clip",
                        "fps": 6,
                        "target_fps": 6,
                        "segmentation_mode": "semantic",
                    }
                ]
            },
        ),
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["source"] == "video", body["source"]
    assert body["frame_count"] == 12, body["frame_count"]
    assert body["frame_names"][0] == "00000000.jpg"
    assert body["frame_names"][-1] == "00000011.jpg"
    assert body["original_fps"] == 25.0, body["original_fps"]
    assert (body["width"], body["height"]) == (320, 240)
    assert body["mode"] == "semantic"
    assert body["video_entry"] == "video/clip.mp4"
    ok("POST /api/sessions decodes a video archive into 8-digit frames")

    frame = client.get(f"/api/sessions/{body['session_id']}/frames/11")
    assert frame.status_code == 200
    assert frame.content[:2] == b"\xff\xd8", "not a JPEG"
    client.delete(f"/api/sessions/{body['session_id']}")
    ok("a frame from a video session is served as a JPEG")


def test_drift_is_reported(client, video_path):
    response = post(
        client,
        "drift.project",
        video_archive(
            video_path,
            {
                "videos": [
                    {
                        "video_name": "clip",
                        "target_fps": 6,
                        "file_names": ["00000000.jpg"],
                    }
                ]
            },
        ),
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["frame_names_match"] is False, body
    assert body["recorded_frame_names"] == ["00000000.jpg"]
    assert body["frame_count"] == 12
    client.delete(f"/api/sessions/{body['session_id']}")
    ok("a recorded list that disagrees with the frames is surfaced in the response")


def test_failures_do_not_leak_sessions(client):
    before = len(session_dirs())

    not_zip = post(client, "broken.project", b"definitely not a zip")
    assert not_zip.status_code == 415, not_zip.status_code
    ok("a non-ZIP upload is 415")

    empty = post(
        client,
        "empty.project",
        frames_archive({}, {"videos": [{}]}),
    )
    assert empty.status_code == 422, empty.status_code
    assert "nothing to review" in empty.json()["detail"], empty.json()
    ok("an archive with neither frames nor a video is 422 with a clear message")

    assert len(session_dirs()) == before, "a failed build leaked a session directory"
    ok("a session that fails to build is not left on disk")

    # Ids that cannot be a session id are a 404, never a filesystem lookup. The
    # guard itself is unit-tested with "../../etc" in test_sessions.py; httpx
    # normalises encoded separators out of the URL before it is ever sent.
    assert client.get("/api/sessions/deadbeef").status_code == 404
    assert client.get("/api/sessions/zzzz").status_code == 404, "non-hex id"
    long_id = "0" * 65
    assert client.get(f"/api/sessions/{long_id}").status_code == 404, "over-long id"
    assert client.delete("/api/sessions/zzzz").status_code == 204
    ok("unknown and malformed session ids cannot reach the filesystem")


def test_startup_sweep(video_path):
    """A session left idle by a crashed run is reaped when the app starts."""
    import time

    from src.core.sessions import create_session

    stale = create_session(TEMP_DIR)
    # Well past the configured TTL (6 h), which the sweeper honours — a session
    # idled for anything less survives, including across a restart.
    old = time.time() - 100_000
    os.utime(stale.root, (old, old))
    assert os.path.isdir(stale.root)

    with TestClient(app) as fresh_client:
        assert not os.path.isdir(stale.root), "the stale session survived startup"
        assert fresh_client.get("/health").json() == {"status": "ok"}
    ok("lifespan sweeps a stale session on the way up")


def build_video(path):
    subprocess.run(
        [
            FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=2",
            "-pix_fmt", "yuv420p", path,
        ],
        check=True,
    )
    return path


def main() -> int:
    os.makedirs(TEMP_DIR, exist_ok=True)
    video_path = build_video(os.path.join(ROOT, "test.mp4"))

    try:
        with TestClient(app) as client:
            print("frames project")
            test_frames_round_trip(client)
            print("video project")
            test_video_round_trip(client, video_path)
            print("drift")
            test_drift_is_reported(client, video_path)
            print("failure handling")
            test_failures_do_not_leak_sessions(client)
        print("startup sweep")
        test_startup_sweep(video_path)
    except AssertionError as failure:
        print(f"\nFAILED: {failure}")
        print(f"artifacts kept in {ROOT}")
        return 1

    shutil.rmtree(ROOT, ignore_errors=True)
    print(f"\nALL {checks} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
