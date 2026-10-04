"""HTTP checks for `/api/sessions`, using FastAPI's own test client.

The environment (temp dir, ffmpeg paths, SAM 3 off) is configured by the shared
`conftest` before the app is imported, because `get_settings` is cached and `main`
resolves settings while it is being imported. The `api_client` fixture supplies a
client that already carries the `X-Vsr-Client` header every session endpoint
requires, and `jpeg`/`archive` build the inputs.
"""

import os
import time

from fastapi.testclient import TestClient
from src.core.config import get_settings
from src.core.sessions import create_session
from src.main import app


def post(client, name, data):
    """Upload `data` as a named `.project` archive."""
    return client.post(
        "/api/sessions",
        files={"project": (name, data, "application/zip")},
    )


def session_dirs():
    """The session ids on disk, in the store the running app is using.

    Read from settings rather than a module constant: the directory is the app's
    decision, and a test that hard-codes its own copy stops noticing if it moves.
    """
    root = os.path.join(get_settings().temp_dir, "sessions")
    if not os.path.isdir(root):
        return []
    return sorted(os.listdir(root))


def test_frames_round_trip(api_client, jpeg, archive):
    """A frame folder goes in, the same bytes come back out, in order."""
    # Spelled out rather than left to the fixture's defaults: the response reports
    # the frame size, so the size the test sends is part of what it is asserting.
    frames = {
        "b.jpg": jpeg(width=48, height=32, colour=(10, 200, 10)),
        "a.jpg": jpeg(width=48, height=32, colour=(200, 10, 10)),
        "c.jpg": jpeg(width=48, height=32, colour=(10, 10, 200)),
    }
    response = post(api_client, "clip_007.project", archive(frames=frames))
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

    session_id = body["session_id"]
    for index, name in enumerate(["b.jpg", "a.jpg", "c.jpg"]):
        frame = api_client.get(f"/api/sessions/{session_id}/frames/{index}")
        assert frame.status_code == 200, frame.text
        assert frame.content == frames[name], f"frame {index} bytes differ"
        assert frame.headers["content-type"] == "image/jpeg"
        assert "immutable" in frame.headers["cache-control"]

    again = api_client.get(f"/api/sessions/{session_id}")
    assert again.status_code == 200
    assert again.json()["frame_names"] == ["b.jpg", "a.jpg", "c.jpg"]

    missing = api_client.get(f"/api/sessions/{session_id}/frames/99")
    assert missing.status_code == 404, missing.status_code
    negative = api_client.get(f"/api/sessions/{session_id}/frames/-1")
    assert negative.status_code == 422, negative.status_code

    assert api_client.delete(f"/api/sessions/{session_id}").status_code == 204
    assert session_id not in session_dirs()
    assert api_client.delete(f"/api/sessions/{session_id}").status_code == 204


def test_deleted_frame_does_not_shift_later_indices(api_client, jpeg, archive):
    """Losing one frame must not renumber the rest of the clip.

    The symptom of index drift is the worst kind: the advertised list shrinks,
    so a frame that still exists becomes unreachable — and with a hole earlier
    in the clip, `GET .../frames/{i}` answers 200 with a *different* frame's
    bytes, quietly showing the reviewer the wrong image. The hole belongs at its
    own index as a 404, with every other index unmoved.
    """
    frames = {
        "a.jpg": jpeg(colour=(200, 0, 0)),
        "b.jpg": jpeg(colour=(0, 200, 0)),
        "c.jpg": jpeg(colour=(0, 0, 200)),
    }
    opened = post(api_client, "hole.project", archive(frames=frames))
    assert opened.status_code == 201, opened.text
    session_id = opened.json()["session_id"]

    # Remove the middle frame behind the API's back, as a partial copy would.
    frames_dir = os.path.join(get_settings().temp_dir, "sessions", session_id, "frames")
    os.remove(os.path.join(frames_dir, "b.jpg"))

    described = api_client.get(f"/api/sessions/{session_id}")
    assert described.status_code == 200
    assert described.json()["frame_names"] == ["a.jpg", "b.jpg", "c.jpg"]
    assert described.json()["frame_count"] == 3

    get = lambda index: api_client.get(f"/api/sessions/{session_id}/frames/{index}")
    assert get(0).content == frames["a.jpg"]
    assert get(1).status_code == 404
    assert get(2).content == frames["c.jpg"], "index 2 served the wrong frame"

    api_client.delete(f"/api/sessions/{session_id}")


def test_video_round_trip(api_client, video, archive):
    response = post(
        api_client,
        "from_video.project",
        archive(
            video=video,
            dataset={
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

    frame = api_client.get(f"/api/sessions/{body['session_id']}/frames/11")
    assert frame.status_code == 200
    assert frame.content[:2] == b"\xff\xd8", "not a JPEG"
    api_client.delete(f"/api/sessions/{body['session_id']}")


def test_drift_is_reported(api_client, video, archive):
    response = post(
        api_client,
        "drift.project",
        archive(
            video=video,
            dataset={
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
    api_client.delete(f"/api/sessions/{body['session_id']}")


def test_failures_do_not_leak_sessions(api_client, archive):
    before = len(session_dirs())

    not_zip = post(api_client, "broken.project", b"definitely not a zip")
    assert not_zip.status_code == 415, not_zip.status_code

    empty = post(
        api_client,
        "empty.project",
        archive(frames={}, dataset={"videos": [{}]}),
    )
    assert empty.status_code == 422, empty.status_code
    assert "nothing to review" in empty.json()["detail"], empty.json()

    # A build that fails must not leave its half-made session behind, or a
    # client could fill the disk with failed uploads.
    assert len(session_dirs()) == before, "a failed build leaked a session directory"

    # Ids that cannot be a session id are a 404, never a filesystem lookup. The
    # guard itself is unit-tested with "../../etc" in test_sessions.py; httpx
    # normalises encoded separators out of the URL before it is ever sent.
    assert api_client.get("/api/sessions/deadbeef").status_code == 404
    assert api_client.get("/api/sessions/zzzz").status_code == 404, "non-hex id"
    long_id = "0" * 65
    assert api_client.get(f"/api/sessions/{long_id}").status_code == 404, "over-long id"
    assert api_client.delete("/api/sessions/zzzz").status_code == 204


def test_client_scoping(api_client, jpeg, archive):
    """The client id is required, and it is what makes a session visible.

    Without this, `GET /api/sessions/{id}/frames/{n}` is readable by anyone who
    learns an id — and the propagation job list hands out ids.
    """
    frames = {"a.jpg": jpeg(), "b.jpg": jpeg(colour=(10, 200, 10))}
    opened = post(api_client, "scoped.project", archive(frames=frames))
    assert opened.status_code == 201, opened.text
    session_id = opened.json()["session_id"]

    assert api_client.get(f"/api/sessions/{session_id}").status_code == 200
    assert api_client.get(f"/api/sessions/{session_id}/frames/0").status_code == 200

    # No header at all is refused, and the message says what to send.
    anonymous = TestClient(app)
    refused = anonymous.get(f"/api/sessions/{session_id}")
    assert refused.status_code == 422, refused.status_code
    assert "X-Vsr-Client" in refused.json()["detail"], refused.json()
    assert anonymous.get(f"/api/sessions/{session_id}/frames/0").status_code == 422

    # A malformed one is refused too, so a guessable id cannot be substituted.
    for bad in ("", "user1", "z" * 32, "a" * 31):
        probe = TestClient(app)
        probe.headers["X-Vsr-Client"] = bad
        assert probe.get(f"/api/sessions/{session_id}").status_code == 422, bad

    # A different client sees nothing, and cannot learn that the session exists.
    stranger = TestClient(app)
    stranger.headers["X-Vsr-Client"] = "b" * 32
    assert stranger.get(f"/api/sessions/{session_id}").status_code == 404
    assert stranger.get(f"/api/sessions/{session_id}/frames/0").status_code == 404
    # A stranger's delete is a no-op, not a way to drop someone else's frames.
    assert stranger.delete(f"/api/sessions/{session_id}").status_code == 204
    assert (
        api_client.get(f"/api/sessions/{session_id}").status_code == 200
    ), "a stranger's delete removed another client's session"
    assert api_client.get(f"/api/sessions/{session_id}/frames/0").status_code == 200


def test_startup_sweep(client_id):
    """A session left idle by a crashed run is reaped when the app starts."""
    stale = create_session(get_settings().temp_dir, owner=client_id)
    # Well past the configured TTL (6 h), which the sweeper honours — a session
    # idled for anything less survives, including across a restart.
    old = time.time() - 100_000
    os.utime(stale.root, (old, old))
    assert os.path.isdir(stale.root)

    with TestClient(app) as fresh_client:
        assert not os.path.isdir(stale.root), "the stale session survived startup"
        assert fresh_client.get("/health").json() == {"status": "ok"}
