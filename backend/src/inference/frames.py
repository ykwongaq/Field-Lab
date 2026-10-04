"""Reading a session's frames for the models.

Both models want decoded RGB images, and both want them for a bounded set of
indices (one frame for a prompt, one window for propagation), so this module is
the single place that knows how to turn a `Session` plus frame indices into PIL
images. Keeping it separate means the services never touch paths, and the
window size is the only thing that decides how much memory a run costs.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

from PIL import Image

from src.core.errors import InvalidRequest
from src.core.sessions import Session

#: Guard against loading a frame whose header is absurd, in pixels.
MAX_PIXELS = 64_000_000


def load_frame(session: Session, index: int) -> Image.Image:
    """One frame as RGB, ready for a model."""
    return decode(session.resolve_frame(index), index)


def load_frames(session: Session, indices: Iterable[int]) -> List[Image.Image]:
    """Several frames as RGB, in the order given."""
    return [load_frame(session, index) for index in indices]


def frame_size(session: Session, index: int = 0) -> tuple:
    """`(width, height)` of a frame, read from its header."""
    path = session.resolve_frame(index)
    try:
        with Image.open(path) as image:
            return image.size
    except OSError as exc:
        raise InvalidRequest(f"Frame {index} could not be read: {exc}") from exc


def decode(path: str, index: int) -> Image.Image:
    """Read an image file as RGB, re-porting an unreadable file as a 422."""
    try:
        with Image.open(path) as image:
            if image.width * image.height > MAX_PIXELS:
                raise InvalidRequest(
                    f"Frame {index} is {image.width}x{image.height} px, which is "
                    "larger than this backend accepts."
                )
            return image.convert("RGB")
    except OSError as exc:
        raise InvalidRequest(f"Frame {index} could not be decoded: {exc}") from exc


def uniform_size(images: Sequence[Image.Image]) -> tuple:
    """`(width, height)`, refusing a set of frames that do not agree.

    The tracker builds one feature pyramid for the whole window, so a window of
    mixed sizes cannot be propagated. Sessions are single-source, so this only
    trips on a corrupt archive.
    """
    if not images:
        raise InvalidRequest("No frames were given.")
    width, height = images[0].size
    for position, image in enumerate(images):
        if image.size != (width, height):
            raise InvalidRequest(
                f"Frame {position} of the window is {image.width}x{image.height} px "
                f"but the first is {width}x{height}; a propagation window must be "
                "a single frame size."
            )
    return (width, height)
