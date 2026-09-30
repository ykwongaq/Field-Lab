"""Checks for the `.project` archive layout contract and the seam it forms.

`projects.layout` exists so that the *reader* does not depend on the *writer*.
That is an architectural claim about imports, which is the kind of thing that
rots back silently, so it is asserted here rather than merely intended.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.core.errors import InvalidRequest
from src.projects import layout

BACKEND_ROOT = Path(__file__).resolve().parents[1]


def test_archive_frames_are_six_digits():
    """The archive contract, which is *not* the session one (8 digits)."""
    assert layout.archive_frame_name(0) == "000000.jpg"
    assert layout.archive_frame_name(11) == "000011.jpg"
    assert layout.archive_frame_name(123456) == "123456.jpg"
    assert layout.ARCHIVE_FRAME_PATTERN.startswith("{:06d}")


def test_natural_key_orders_numerically():
    names = ["frame_10.jpg", "frame_2.jpg", "frame_1.jpg"]
    assert sorted(names, key=layout.natural_key) == [
        "frame_1.jpg",
        "frame_2.jpg",
        "frame_10.jpg",
    ]


def test_sanitize_name_makes_a_safe_component():
    assert layout.sanitize_name("clip 007") == "clip_007"
    assert layout.sanitize_name("../../etc/passwd") == "etc_passwd"
    # An empty or all-punctuation name still has to yield something usable.
    assert layout.sanitize_name("   ") == "project"
    assert layout.sanitize_name("...", fallback="clip") == "clip"


def test_project_name_prefers_the_callers_name():
    assert layout.project_name_for("my clip", "ignored.mp4") == "my_clip"
    assert layout.project_name_for("", "/tmp/some clip.mp4") == "some_clip"
    assert layout.project_name_for(None, "clip.mp4") == "clip"


def test_select_frame_files_orders_and_filters(scratch):
    """Only images, in natural order — a stray text file must not be a frame."""
    folder = os.path.join(str(scratch), "frames")
    os.makedirs(folder)
    for name in ("f10.jpg", "f2.jpg", "f1.png", "notes.txt", "f3.JPEG"):
        with open(os.path.join(folder, name), "wb") as handle:
            handle.write(b"x")

    assert layout.select_frame_files(folder) == [
        "f1.png",
        "f2.jpg",
        "f3.JPEG",
        "f10.jpg",
    ]


def test_select_frame_files_rejects_an_empty_folder(scratch):
    folder = os.path.join(str(scratch), "empty")
    os.makedirs(folder)
    with pytest.raises(InvalidRequest, match="holds no images"):
        layout.select_frame_files(folder)
    with pytest.raises(InvalidRequest, match="not found"):
        layout.select_frame_files(os.path.join(str(scratch), "nowhere"))


def test_the_reader_does_not_import_the_writer():
    """`loader` must not drag in `builder` — the module that owns video encoding.

    Checked in a subprocess because the test process has already imported half the
    package; only a fresh interpreter can answer what importing `loader` pulls in.
    """
    code = (
        "import sys\n"
        "import src.projects.loader\n"
        "leaked = [name for name in ('src.projects.builder', 'cv2') if name in sys.modules]\n"
        "print(','.join(leaked))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(BACKEND_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "", (
        "importing the archive reader pulled in the writer: " + result.stdout.strip()
    )


def test_the_layout_module_is_a_leaf():
    """It may depend on `core.errors`, and on nothing else of ours."""
    code = (
        "import sys\n"
        "import src.projects.layout\n"
        "leaked = [name for name in ('cv2', 'torch', 'numpy', 'src.projects.builder',\n"
        "                           'src.projects.loader', 'src.domain', 'src.inference')\n"
        "          if name in sys.modules]\n"
        "print(','.join(leaked))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(BACKEND_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "", result.stdout.strip()
