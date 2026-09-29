"""Session store and ffmpeg extraction: integration checks on real tooling.

Run from the backend root::

    python test/test_sessions.py

ffmpeg/ffprobe are taken from PATH, or from ``FFMPEG_BIN`` / ``FFPROBE_BIN``
when they are not installed globally. Everything is written under a throwaway
directory, which is removed on success and kept for inspection on failure.
"""

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_ROOT)

from PIL import Image  # noqa: E402

from src.core.config import Settings  # noqa: E402
from src.core.errors import (  # noqa: E402
    InvalidRequest,
    NotFound,
    UnsupportedMediaType,
)
from src.core.sessions import (  # noqa: E402
    create_session,
    delete_session,
    frame_index,
    frame_name,
    list_frame_names,
    open_session,
    sweep_sessions,
)
from src.domain import extract  # noqa: E402
from src.projects import loader  # noqa: E402

FFMPEG = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg")
FFPROBE = os.environ.get("FFPROBE_BIN") or shutil.which("ffprobe")
assert FFMPEG and os.path.isfile(FFMPEG), f"ffmpeg not found: {FFMPEG!r}"
assert FFPROBE and os.path.isfile(FFPROBE), f"ffprobe not found: {FFPROBE!r}"

checks = 0


def ok(label):
    global checks
    checks += 1
    print("  ok:", label)


def settings_for(root: str) -> Settings:
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
    ok("frame_name/frame_index round-trip (8 digits)")


def test_ordering(tmp):
    frames = os.path.join(tmp, "order", "frames")
    os.makedirs(frames, exist_ok=True)
    for name in ("00000010.jpg", "00000002.jpg", "00000001.jpg", "notes.txt"):
        with open(os.path.join(frames, name), "wb") as handle:
            handle.write(b"x")
    assert list_frame_names(frames) == [
        "00000001.jpg",
        "00000002.jpg",
        "00000010.jpg",
    ], list_frame_names(frames)
    ok("list_frame_names orders numerically and skips non-images")


def test_session_lifecycle(tmp):
    session = create_session(tmp, meta={"name": "clip_007", "mode": "instance"})
    assert os.path.isdir(session.frames_dir)
    assert os.path.isdir(session.source_dir)
    assert session.read_meta()["name"] == "clip_007"
    assert session.frame_count() == 0
    assert session.frame_path(3).endswith(os.path.join("frames", "00000003.jpg"))
    ok("create_session lays out frames/ + source/ + session.json")

    try:
        session.frame_path(-1)
    except InvalidRequest:
        ok("negative frame index is rejected")
    else:
        raise AssertionError("negative index accepted")

    with open(session.frame_path(0), "wb") as handle:
        handle.write(b"x" * 10)
    assert session.frame_count() == 1
    ok("frame_count reflects what is on disk")

    reopened = open_session(tmp, session.id)
    assert reopened.id == session.id and reopened.root == session.root
    ok("open_session finds it again")

    try:
        open_session(tmp, "../../etc")
    except NotFound:
        ok("path traversal in a session id is rejected")
    else:
        raise AssertionError("traversal accepted")

    try:
        open_session(tmp, "0" * 64 + "f")
    except NotFound:
        ok("over-long session id is rejected")
    else:
        raise AssertionError("over-long id accepted")


def test_touch_and_ttl(tmp):
    stale = create_session(tmp)
    fresh = create_session(tmp)
    old_stamp = time.time() - 10_000
    os.utime(stale.root, (old_stamp, old_stamp))

    removed = sweep_sessions(tmp, ttl_seconds=3600)
    assert removed == [stale.id], removed
    assert not os.path.isdir(stale.root)
    assert os.path.isdir(fresh.root)
    ok("sweep_sessions drops an idle session and keeps a fresh one")

    # `ttl_seconds=0` means "remove anything strictly older than now". A session
    # created a moment ago may carry an mtime at or after `time.time()` (Windows
    # ticks are coarse), so age it deliberately first.
    aged = time.time() - 5
    os.utime(fresh.root, (aged, aged))
    removed = sweep_sessions(tmp, ttl_seconds=0)
    assert fresh.id in removed, removed
    ok("sweep_sessions with ttl 0 clears the rest")


def test_quota(tmp):
    sessions = []
    base = time.time() - 3000
    for index in range(3):
        session = create_session(tmp, meta={"i": index})
        with open(os.path.join(session.frames_dir, "00000000.jpg"), "wb") as handle:
            handle.write(b"x" * 2000)
        stamp = base + index * 100
        os.utime(session.root, (stamp, stamp))
        sessions.append(session)

    removed = sweep_sessions(tmp, ttl_seconds=10**9, max_bytes=2500)
    assert sessions[0].id in removed, removed
    assert not os.path.isdir(sessions[0].root)
    assert os.path.isdir(sessions[2].root)
    ok("quota sweep evicts least-recently-used first")

    assert delete_session(tmp, sessions[2].id) is True
    assert delete_session(tmp, sessions[2].id) is False
    ok("delete_session is idempotent")


def test_probe(video):
    info = extract.probe_video(video, ffprobe=FFPROBE)
    assert info.fps == 25.0, info
    assert (info.width, info.height) == (320, 240), info
    assert info.duration_seconds == 2.0, info
    assert info.frame_count == 50, info
    ok("probe_video reads fps/size/duration/frame_count from real ffprobe json")


def test_extract(tmp, video):
    session = create_session(tmp, meta={"name": "from_video"})
    names = extract.extract_frames(
        video, session.frames_dir, fps=6, jpeg_quality=95, ffmpeg=FFMPEG
    )
    assert len(names) == 12, names
    assert names[0] == "00000000.jpg", names[0]
    assert names[-1] == "00000011.jpg", names[-1]
    assert session.frame_names() == names
    assert session.frame_count() == 12
    ok("extract_frames yields 12 zero-based frames for 2s at 6fps")

    scratch = os.path.join(os.path.dirname(session.frames_dir), "frames.raw")
    assert not os.path.exists(scratch), scratch
    ok("the scratch directory is cleaned up")

    size = session.size_bytes()
    assert size > 0
    ok(f"session size accounting works ({size} bytes)")

    assert extract.qscale_for(100) == 1
    assert extract.qscale_for(95) == 2
    assert extract.qscale_for(80) == 5
    ok("quality maps onto ffmpeg's inverted qscale")

    totals = {}
    for quality in (95, 80):
        out = os.path.join(tmp, f"q{quality}", "frames")
        extract.extract_frames(
            video, out, fps=6, jpeg_quality=quality, ffmpeg=FFMPEG
        )
        totals[quality] = sum(
            os.path.getsize(os.path.join(out, name))
            for name in list_frame_names(out)
        )
        ok(
            f"quality {quality} (q:v {extract.qscale_for(quality)}) "
            f"-> {totals[quality]} bytes over 12 frames"
        )
    assert totals[80] < totals[95], totals
    ok("lower quality yields a smaller frame set")


def test_reproducible(tmp, video):
    """The session model leans on re-extracting the same video giving the same
    sequence, so this is a correctness check rather than a performance one."""
    runs = {}
    for label in ("run1", "run2"):
        out = os.path.join(tmp, label, "frames")
        runs[label] = (
            out,
            extract.extract_frames(
                video, out, fps=6, jpeg_quality=95, ffmpeg=FFMPEG
            ),
        )

    (out_a, names_a), (out_b, names_b) = runs.values()
    assert names_a == names_b, (names_a[:3], names_b[:3])
    ok(f"both runs produce the same {len(names_a)} names")

    def digests(out_dir):
        result = []
        for name in list_frame_names(out_dir):
            with open(os.path.join(out_dir, name), "rb") as handle:
                result.append(hashlib.sha1(handle.read()).hexdigest())
        return result

    digests_a, digests_b = digests(out_a), digests(out_b)
    assert digests_a == digests_b, "frame bytes differ between runs"
    ok("both runs are byte-identical on this ffmpeg build")


def test_renumber(tmp):
    """The safeguard: non-zero-based input is normalised."""
    raw = os.path.join(tmp, "renumber", "raw")
    out = os.path.join(tmp, "renumber", "frames")
    os.makedirs(raw, exist_ok=True)
    os.makedirs(out, exist_ok=True)
    for name in ("00000001.jpg", "00000002.jpg", "00000003.jpg"):
        with open(os.path.join(raw, name), "wb") as handle:
            handle.write(b"x")

    written = extract._renumber(raw, out, list_frame_names(raw))
    assert written == ["00000000.jpg", "00000001.jpg", "00000002.jpg"], written
    assert list_frame_names(out) == written
    ok("_renumber normalises an ffmpeg run that started at 1")


def test_missing_binaries():
    assert extract.missing_binaries(FFMPEG, FFPROBE) == []
    absent = extract.missing_binaries("no-such-ffmpeg-xyz", "no-such-ffprobe-xyz")
    assert absent == ["no-such-ffmpeg-xyz", "no-such-ffprobe-xyz"], absent
    ok("missing_binaries accepts real tooling and reports absent ones")


def test_bad_input(tmp):
    try:
        extract.extract_frames(
            os.path.join(tmp, "missing.mp4"),
            os.path.join(tmp, "nowhere"),
            fps=6,
            ffmpeg=FFMPEG,
        )
    except InvalidRequest:
        ok("a missing video is rejected before ffmpeg runs")
    else:
        raise AssertionError("missing video accepted")

    try:
        extract.qscale_for(0)
    except InvalidRequest:
        ok("an out-of-range JPEG quality is rejected")
    else:
        raise AssertionError("quality 0 accepted")


def test_loader_frames(tmp):
    """An archive carrying frames is copied in, keeping the recorded order."""
    settings = settings_for(tmp)
    path = make_archive(
        os.path.join(tmp, "frames.project"),
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
    ok("a frames/ archive is copied in, in the order the annotation recorded")


def test_loader_frames_without_record(tmp):
    """With no recorded order, entry names are sorted naturally."""
    settings = settings_for(tmp)
    path = make_archive(
        os.path.join(tmp, "unordered.project"),
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
    ok("frames without a recorded order fall back to natural sorting")


def test_loader_video(tmp, video):
    """An archive carrying a video has its frames decoded by ffmpeg."""
    settings = settings_for(tmp)
    with open(video, "rb") as handle:
        data = handle.read()

    path = make_archive(
        os.path.join(tmp, "video.project"),
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
    ok("a video/ archive is decoded into 8-digit frames at the requested fps")


def test_loader_drift(tmp, video):
    """A recorded frame list that disagrees with the extraction is reported."""
    settings = settings_for(tmp)
    with open(video, "rb") as handle:
        data = handle.read()

    path = make_archive(
        os.path.join(tmp, "drift.project"),
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
    ok("a recorded frame list that disagrees with the frames is flagged")


def test_loader_frames_match_when_video_is_unchanged(tmp, video):
    """The same extraction recorded in the archive reports no drift."""
    settings = settings_for(tmp)
    with open(video, "rb") as handle:
        data = handle.read()

    expected = [frame_name(i) for i in range(12)]
    path = make_archive(
        os.path.join(tmp, "stable.project"),
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
    ok("a re-extraction that reproduces the recorded list reports no drift")


def test_loader_bad_input(tmp):
    settings = settings_for(tmp)

    empty = make_archive(os.path.join(tmp, "empty.project"), dataset={"videos": [{}]})
    try:
        loader.load_archive_into_session(
            empty, create_session(settings.temp_dir), settings
        )
    except InvalidRequest:
        ok("an archive with neither frames nor a video is rejected")
    else:
        raise AssertionError("an empty archive was accepted")

    not_zip = os.path.join(tmp, "not-a-zip.project")
    with open(not_zip, "wb") as handle:
        handle.write(b"definitely not a zip")
    try:
        loader.load_archive_into_session(
            not_zip, create_session(settings.temp_dir), settings
        )
    except UnsupportedMediaType:
        ok("an upload that is not a ZIP is rejected")
    else:
        raise AssertionError("a non-zip upload was accepted")

    without_annotation = make_archive(
        os.path.join(tmp, "bare.project")
    )
    session = create_session(settings.temp_dir)
    os.makedirs(session.frames_dir, exist_ok=True)
    try:
        loader.load_archive_into_session(without_annotation, session, settings)
    except InvalidRequest:
        ok("an archive with no annotation at all is still rejected cleanly")
    else:
        raise AssertionError("an annotation-less archive was accepted")


def scratch(root: str, name: str) -> str:
    """A clean sub-directory per section, so sections cannot interfere."""
    path = os.path.join(root, name)
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    return path


def run_checks(root: str) -> None:
    print("naming")
    test_naming()
    print("frame ordering")
    test_ordering(scratch(root, "ordering"))
    print("session lifecycle")
    test_session_lifecycle(scratch(root, "lifecycle"))
    print("cleanup")
    test_touch_and_ttl(scratch(root, "ttl"))
    test_quota(scratch(root, "quota"))

    work = scratch(root, "extraction")
    video = os.path.join(work, "test.mp4")
    subprocess.run(
        [
            FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=2",
            "-pix_fmt", "yuv420p", video,
        ],
        check=True,
    )
    print("probe")
    test_probe(video)
    print("extraction")
    test_extract(work, video)
    print("reproducibility")
    test_reproducible(work, video)
    print("renumber safeguard")
    test_renumber(work)
    print("bad input")
    test_bad_input(work)
    print("tooling")
    test_missing_binaries()

    print("loader")
    test_loader_frames(scratch(root, "load-frames"))
    test_loader_frames_without_record(scratch(root, "load-unordered"))
    test_loader_video(scratch(root, "load-video"), video)
    test_loader_drift(scratch(root, "load-drift"), video)
    test_loader_frames_match_when_video_is_unchanged(scratch(root, "load-stable"), video)
    test_loader_bad_input(scratch(root, "load-bad"))


def main() -> int:
    root = tempfile.mkdtemp(prefix="vsr-sessions-")
    try:
        run_checks(root)
    except AssertionError as failure:
        print(f"\nFAILED: {failure}")
        print(f"artifacts kept in {root}")
        return 1
    shutil.rmtree(root, ignore_errors=True)
    print(f"\nALL {checks} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
