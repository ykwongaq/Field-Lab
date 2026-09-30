"""Session store and ffmpeg extraction: integration checks on real tooling.

ffmpeg/ffprobe are taken from PATH, or from ``FFMPEG_BIN`` / ``FFPROBE_BIN`` when
they are not installed globally (the shared `conftest` exports both names). Tests
that actually invoke them declare the `ffmpeg` fixture, so a machine without the
tooling skips them instead of failing the suite.
"""

import functools
import hashlib
import io
import json
import os
import shutil
import time
import zipfile

import pytest
from PIL import Image

from src.core.config import Settings
from src.core.errors import (
    InvalidRequest,
    NotFound,
    UnsupportedMediaType,
)
from src.core.sessions import (
    create_session as _create_session,
    delete_session as _delete_session,
    frame_index,
    frame_name,
    list_frame_names,
    open_session as _open_session,
    sweep_sessions,
)
from src.domain import extract
from src.projects import loader

FFMPEG = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg")
FFPROBE = os.environ.get("FFPROBE_BIN") or shutil.which("ffprobe")

#: Every session belongs to a client. This file's tests all act as one client, so
#: the three owner-taking calls are bound to it once here rather than at each of
#: the twenty-odd call sites. The isolation checks below use the raw `_`-prefixed
#: names explicitly, which is why they are imported that way.
OWNER = "a" * 32
create_session = functools.partial(_create_session, owner=OWNER)
open_session = functools.partial(_open_session, owner=OWNER)
delete_session = functools.partial(_delete_session, owner=OWNER)


def settings_for(root) -> Settings:
    """Settings pointing at a scratch temp dir and the discovered ffmpeg."""
    return Settings(
        temp_dir=os.path.join(root, "sessions-root"),
        frames_jpeg_quality=95,
        ffmpeg_bin=FFMPEG,
        ffprobe_bin=FFPROBE,
    )


def jpeg(width=64, height=48, colour=(120, 40, 200)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buffer, format="JPEG")
    return buffer.getvalue()


def make_archive(path, *, frames=None, video=None, dataset=None) -> str:
    """Build a `.project` by hand, so the loader is tested without the writer."""
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in (frames or {}).items():
            archive.writestr(f"frames/{name}", data)
        if video is not None:
            archive.writestr("video/clip.mp4", video)
        if dataset is not None:
            archive.writestr("annotation.json", json.dumps(dataset))
    return path


def test_naming():
    assert frame_name(0) == "00000000.jpg"
    assert frame_name(11) == "00000011.jpg"
    assert frame_name(12345678) == "12345678.jpg"
    assert frame_index("00000007.jpg") == 7
    assert frame_index("nope.jpg") == -1


def test_ordering(scratch):
    frames = os.path.join(scratch, "order", "frames")
    os.makedirs(frames, exist_ok=True)
    for name in ("00000010.jpg", "00000002.jpg", "00000001.jpg", "notes.txt"):
        with open(os.path.join(frames, name), "wb") as handle:
            handle.write(b"x")
    assert list_frame_names(frames) == [
        "00000001.jpg",
        "00000002.jpg",
        "00000010.jpg",
    ], list_frame_names(frames)


def test_session_lifecycle(scratch):
    session = create_session(scratch, meta={"name": "clip_007", "mode": "instance"})
    assert os.path.isdir(session.frames_dir)
    assert os.path.isdir(session.source_dir)
    assert session.read_meta()["name"] == "clip_007"
    assert session.frame_count() == 0
    assert session.frame_path(3).endswith(os.path.join("frames", "00000003.jpg"))

    with pytest.raises(InvalidRequest):
        session.frame_path(-1)

    with open(session.frame_path(0), "wb") as handle:
        handle.write(b"x" * 10)
    assert session.frame_count() == 1

    reopened = open_session(scratch, session.id)
    assert reopened.id == session.id and reopened.root == session.root

    # A session id is joined to a path, so it is validated rather than trusted.
    with pytest.raises(NotFound):
        open_session(scratch, "../../etc")
    with pytest.raises(NotFound):
        open_session(scratch, "0" * 64 + "f")


def test_ownership_isolates_sessions(scratch):
    """A session is invisible to a client that did not create it."""
    alice, bob = "a" * 32, "b" * 32
    mine = _create_session(scratch, owner=alice)
    theirs = _create_session(scratch, owner=bob)
    assert mine.owner == alice and theirs.owner == bob

    assert _open_session(scratch, mine.id, owner=alice).id == mine.id
    assert _open_session(scratch, theirs.id, owner=bob).id == theirs.id
    # Every owner other than Alice is refused her session, including one that
    # opens Bob's own session just above.
    for intruder in (bob, "", "c" * 32):
        with pytest.raises(NotFound):
            _open_session(scratch, mine.id, owner=intruder)

    # Deleting is scoped the same way, so one client cannot drop another's work.
    assert _delete_session(scratch, theirs.id, owner=alice) is False
    assert os.path.isdir(theirs.root)
    assert _delete_session(scratch, theirs.id, owner=bob) is True


def test_quota_is_per_owner(scratch):
    """An owner over its budget loses its own sessions, never a colleague's."""
    alice, bob = "a" * 32, "b" * 32
    base = time.time() - 3000

    def pad(session, size, stamp):
        with open(os.path.join(session.frames_dir, "00000000.jpg"), "wb") as handle:
            handle.write(b"x" * size)
        os.utime(session.root, (stamp, stamp))

    alice_old = _create_session(scratch, owner=alice)
    pad(alice_old, 2000, base)
    alice_new = _create_session(scratch, owner=alice)
    pad(alice_new, 2000, base + 100)
    bob_only = _create_session(scratch, owner=bob)
    pad(bob_only, 2000, base + 200)

    # Alice is over 3000 bytes on her own; Bob is not.
    removed = sweep_sessions(scratch, ttl_seconds=10**9, max_bytes_per_owner=3000)
    assert removed == [alice_old.id], removed
    assert os.path.isdir(alice_new.root), "alice lost a session she was still using"
    assert os.path.isdir(bob_only.root), "bob's session was evicted for alice's quota"


def test_touch_and_ttl(scratch):
    stale = create_session(scratch)
    fresh = create_session(scratch)
    old_stamp = time.time() - 10_000
    os.utime(stale.root, (old_stamp, old_stamp))

    removed = sweep_sessions(scratch, ttl_seconds=3600)
    assert removed == [stale.id], removed
    assert not os.path.isdir(stale.root)
    assert os.path.isdir(fresh.root)

    # `ttl_seconds=0` means "remove anything strictly older than now". A session
    # created a moment ago may carry an mtime at or after `time.time()` (Windows
    # ticks are coarse), so age it deliberately first.
    aged = time.time() - 5
    os.utime(fresh.root, (aged, aged))
    removed = sweep_sessions(scratch, ttl_seconds=0)
    assert fresh.id in removed, removed


def test_quota(scratch):
    sessions = []
    base = time.time() - 3000
    for index in range(3):
        session = create_session(scratch, meta={"i": index})
        with open(os.path.join(session.frames_dir, "00000000.jpg"), "wb") as handle:
            handle.write(b"x" * 2000)
        stamp = base + index * 100
        os.utime(session.root, (stamp, stamp))
        sessions.append(session)

    removed = sweep_sessions(scratch, ttl_seconds=10**9, max_bytes=2500)
    assert sessions[0].id in removed, removed
    assert not os.path.isdir(sessions[0].root)
    assert os.path.isdir(sessions[2].root)

    assert delete_session(scratch, sessions[2].id) is True
    assert delete_session(scratch, sessions[2].id) is False


def test_probe(video, ffmpeg):
    info = extract.probe_video(video, ffprobe=FFPROBE)
    assert info.fps == 25.0, info
    assert (info.width, info.height) == (320, 240), info
    assert info.duration_seconds == 2.0, info
    assert info.frame_count == 50, info


def test_extract(scratch, video, ffmpeg):
    session = create_session(scratch, meta={"name": "from_video"})
    names = extract.extract_frames(
        video, session.frames_dir, fps=6, jpeg_quality=95, ffmpeg=FFMPEG
    )
    assert len(names) == 12, names
    assert names[0] == "00000000.jpg", names[0]
    assert names[-1] == "00000011.jpg", names[-1]
    assert session.frame_names() == names
    assert session.frame_count() == 12

    # ffmpeg is told to write into a sibling `frames.raw`, which must be gone once
    # the frames have been renumbered into place.
    staged = os.path.join(os.path.dirname(session.frames_dir), "frames.raw")
    assert not os.path.exists(staged), staged

    size = session.size_bytes()
    assert size > 0

    assert extract.qscale_for(100) == 1
    assert extract.qscale_for(95) == 2
    assert extract.qscale_for(80) == 5

    # Lower quality must buy a smaller frame set, which is the whole reason the
    # setting exists.
    totals = {}
    for quality in (95, 80):
        out = os.path.join(scratch, f"q{quality}", "frames")
        extract.extract_frames(video, out, fps=6, jpeg_quality=quality, ffmpeg=FFMPEG)
        totals[quality] = sum(
            os.path.getsize(os.path.join(out, name)) for name in list_frame_names(out)
        )
    assert totals[80] < totals[95], totals


def test_reproducible(scratch, video, ffmpeg):
    """The session model leans on re-extracting the same video giving the same
    sequence, so this is a correctness check rather than a performance one."""
    runs = {}
    for label in ("run1", "run2"):
        out = os.path.join(scratch, label, "frames")
        runs[label] = (
            out,
            extract.extract_frames(video, out, fps=6, jpeg_quality=95, ffmpeg=FFMPEG),
        )

    (out_a, names_a), (out_b, names_b) = runs.values()
    assert names_a == names_b, (names_a[:3], names_b[:3])

    def digests(out_dir):
        result = []
        for name in list_frame_names(out_dir):
            with open(os.path.join(out_dir, name), "rb") as handle:
                result.append(hashlib.sha1(handle.read()).hexdigest())
        return result

    assert digests(out_a) == digests(out_b), "frame bytes differ between runs"


def test_renumber(scratch):
    """The safeguard: non-zero-based input is normalised."""
    raw = os.path.join(scratch, "renumber", "raw")
    out = os.path.join(scratch, "renumber", "frames")
    os.makedirs(raw, exist_ok=True)
    os.makedirs(out, exist_ok=True)
    for name in ("00000001.jpg", "00000002.jpg", "00000003.jpg"):
        with open(os.path.join(raw, name), "wb") as handle:
            handle.write(b"x")

    written = extract._renumber(raw, out, list_frame_names(raw))
    assert written == ["00000000.jpg", "00000001.jpg", "00000002.jpg"], written
    assert list_frame_names(out) == written


def test_missing_binaries(ffmpeg):
    assert extract.missing_binaries(FFMPEG, FFPROBE) == []
    absent = extract.missing_binaries("no-such-ffmpeg-xyz", "no-such-ffprobe-xyz")
    assert absent == ["no-such-ffmpeg-xyz", "no-such-ffprobe-xyz"], absent


def test_bad_input(scratch, ffmpeg):
    with pytest.raises(InvalidRequest):
        extract.extract_frames(
            os.path.join(scratch, "missing.mp4"),
            os.path.join(scratch, "nowhere"),
            fps=6,
            ffmpeg=FFMPEG,
        )
    with pytest.raises(InvalidRequest):
        extract.qscale_for(0)


def test_loader_frames(scratch):
    """An archive carrying frames is copied in, keeping the recorded order."""
    settings = settings_for(scratch)
    path = make_archive(
        os.path.join(scratch, "frames.project"),
        frames={
            "b.jpg": jpeg(colour=(10, 200, 10)),
            "a.jpg": jpeg(colour=(200, 10, 10)),
            "c.jpg": jpeg(colour=(10, 10, 200)),
        },
        dataset={
            "videos": [
                {
                    "id": 1,
                    "video_name": "clip",
                    "file_names": ["b.jpg", "a.jpg", "c.jpg"],
                    "fps": 6,
                    "segmentation_mode": "instance",
                }
            ],
            "annotations": [],
            "categories": [],
        },
    )
    session = create_session(settings.temp_dir)
    loaded = loader.load_archive_into_session(path, session, settings)

    assert loaded.source == "frames", loaded.source
    assert loaded.frame_names == ["b.jpg", "a.jpg", "c.jpg"], loaded.frame_names
    assert loaded.fps == 6
    assert (loaded.width, loaded.height) == (64, 48)
    assert loaded.mode == "instance"
    assert loaded.frame_names_match is True
    assert session.frame_names() == ["b.jpg", "a.jpg", "c.jpg"]
    assert session.frame_name_at(1) == "a.jpg"
    assert session.read_meta()["source"] == "frames"


def test_loader_frames_without_record(scratch):
    """With no recorded order, entry names are sorted naturally."""
    settings = settings_for(scratch)
    path = make_archive(
        os.path.join(scratch, "unordered.project"),
        frames={
            "frame_10.jpg": jpeg(),
            "frame_2.jpg": jpeg(),
            "frame_1.jpg": jpeg(),
        },
        dataset={"videos": [{}], "annotations": [], "categories": []},
    )
    session = create_session(settings.temp_dir)
    loaded = loader.load_archive_into_session(path, session, settings)
    assert loaded.frame_names == [
        "frame_1.jpg",
        "frame_2.jpg",
        "frame_10.jpg",
    ], loaded.frame_names


def test_loader_video(scratch, video, ffmpeg):
    """An archive carrying a video has its frames decoded by ffmpeg."""
    settings = settings_for(scratch)
    with open(video, "rb") as handle:
        data = handle.read()

    path = make_archive(
        os.path.join(scratch, "video.project"),
        video=data,
        dataset={
            "videos": [
                {
                    "id": 1,
                    "video_name": "clip",
                    "file_names": [],
                    "fps": 6,
                    "target_fps": 6,
                    "segmentation_mode": "semantic",
                }
            ],
            "annotations": [],
            "categories": [],
        },
    )
    session = create_session(settings.temp_dir)
    loaded = loader.load_archive_into_session(path, session, settings)

    assert loaded.source == "video", loaded.source
    assert len(loaded.frame_names) == 12, loaded.frame_names
    assert loaded.frame_names[0] == "00000000.jpg"
    assert loaded.frame_names[-1] == "00000011.jpg"
    assert loaded.fps == 6
    assert loaded.original_fps == 25.0, loaded.original_fps
    assert (loaded.width, loaded.height) == (320, 240), (loaded.width, loaded.height)
    assert loaded.mode == "semantic"
    assert loaded.video_entry == "video/clip.mp4"
    assert session.frame_name_at(11) == "00000011.jpg"
    assert session.frame_name_at(99) == "00000099.jpg"  # synthesised fallback
    assert os.path.isfile(os.path.join(session.source_dir, "clip.mp4"))


def test_loader_drift(scratch, video, ffmpeg):
    """A recorded frame list that disagrees with the extraction is reported."""
    settings = settings_for(scratch)
    with open(video, "rb") as handle:
        data = handle.read()

    path = make_archive(
        os.path.join(scratch, "drift.project"),
        video=data,
        dataset={
            "videos": [
                {
                    "video_name": "clip",
                    "target_fps": 6,
                    # Only one frame where the extraction will produce twelve.
                    "file_names": ["00000000.jpg"],
                }
            ]
        },
    )
    session = create_session(settings.temp_dir)
    loaded = loader.load_archive_into_session(path, session, settings)
    assert loaded.frame_names_match is False
    assert session.read_meta()["frame_names_match"] is False


def test_loader_frames_match_when_video_is_unchanged(scratch, video, ffmpeg):
    """The same extraction recorded in the archive reports no drift."""
    settings = settings_for(scratch)
    with open(video, "rb") as handle:
        data = handle.read()

    expected = [frame_name(i) for i in range(12)]
    path = make_archive(
        os.path.join(scratch, "stable.project"),
        video=data,
        dataset={
            "videos": [
                {
                    "video_name": "clip",
                    "target_fps": 6,
                    "file_names": expected,
                }
            ]
        },
    )
    session = create_session(settings.temp_dir)
    loaded = loader.load_archive_into_session(path, session, settings)
    assert loaded.frame_names == expected
    assert loaded.frame_names_match is True


def test_loader_bad_input(scratch):
    settings = settings_for(scratch)

    empty = make_archive(
        os.path.join(scratch, "empty.project"), dataset={"videos": [{}]}
    )
    with pytest.raises(InvalidRequest):
        loader.load_archive_into_session(
            empty, create_session(settings.temp_dir), settings
        )

    not_zip = os.path.join(scratch, "not-a-zip.project")
    with open(not_zip, "wb") as handle:
        handle.write(b"definitely not a zip")
    with pytest.raises(UnsupportedMediaType):
        loader.load_archive_into_session(
            not_zip, create_session(settings.temp_dir), settings
        )

    # An archive with no annotation at all: nothing names a source, so it is
    # rejected as a bad request rather than crashing the loader.
    without_annotation = make_archive(os.path.join(scratch, "bare.project"))
    session = create_session(settings.temp_dir)
    os.makedirs(session.frames_dir, exist_ok=True)
    with pytest.raises(InvalidRequest):
        loader.load_archive_into_session(without_annotation, session, settings)
