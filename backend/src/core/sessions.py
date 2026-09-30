"""Disposable per-review session directories.

Opening a project always produces one of these: a directory under
``<temp_dir>/sessions/<uuid>/`` holding the clip's frames as plain JPEGs. Every
reader then works from a single frame source — the annotation panel, SAM 3 and
propagation — whatever the archive actually carried (a video, a frame folder, or
both), so nothing downstream has to know where the pixels came from.

Layout::

    <temp_dir>/sessions/<uuid>/
        frames/00000000.jpg     the frame sequence, always 8-digit numbered
        source/<name>           the uploaded archive, kept as provenance
        session.json            what the session was built from

A session also records the client that asked for it, and every lookup presents
one (see `core.identity`). That is what keeps one reviewer's clip out of another
reviewer's reach, and the frames on disk are the only user data the backend
holds.

A session is a cache, not a record. The frame sequence can always be rebuilt
from the archive, so an idle session may be swept at any moment without losing
anything a caller cannot regenerate — which is what makes sweeping safe to do
opportunistically rather than only on a clean shutdown.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.core.errors import InvalidRequest, NotFound

SESSIONS_DIRNAME = "sessions"
FRAMES_DIRNAME = "frames"
SOURCE_DIRNAME = "source"
SESSION_META = "session.json"

#: Frame file names: 8 digits, so a clip can hold 100 million frames.
FRAME_NAME_PATTERN = "{:08d}.jpg"

#: Session ids become directory names, so they are restricted to hex and dashes.
SESSION_ID_PATTERN = re.compile(r"\A[0-9a-fA-F-]{1,64}\Z")

#: Extensions treated as frames when a session's directory is read back.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")

#: Sort key for a name that encodes no index: last, then by name.
_NO_INDEX = 1 << 62


def frame_name(index: int) -> str:
    """Canonical name of frame ``index``."""
    return FRAME_NAME_PATTERN.format(index)


def frame_index(name: str) -> int:
    """The index a frame name encodes, or ``-1`` when it encodes none."""
    stem = os.path.splitext(os.path.basename(name))[0]
    return int(stem) if stem.isdigit() else -1


def sessions_root(temp_dir: str) -> str:
    """The directory every session lives in."""
    return os.path.join(temp_dir, SESSIONS_DIRNAME)


def _sort_key(name: str) -> tuple:
    """Order frames numerically, so any numbering reads back in playback order."""
    index = frame_index(name)
    return (index if index >= 0 else _NO_INDEX, name)


def list_frame_names(frames_dir: str) -> List[str]:
    """The frame files of a directory, in playback order."""
    if not os.path.isdir(frames_dir):
        return []
    names = [
        name
        for name in os.listdir(frames_dir)
        if os.path.isfile(os.path.join(frames_dir, name))
        and os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
    ]
    names.sort(key=_sort_key)
    return names


def _write_json_atomic(path: str, payload: Any) -> None:
    """Write JSON so a reader never sees a half-written file."""
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def directory_size(path: str) -> int:
    """Total size of the files under ``path``, in bytes."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            with contextlib.suppress(OSError):
                total += os.path.getsize(os.path.join(root, name))
    return total


@dataclass
class Session:
    """One open review: a frame working copy plus what it was built from."""

    id: str
    root: str

    @property
    def frames_dir(self) -> str:
        """The frame sequence every reader uses."""
        return os.path.join(self.root, FRAMES_DIRNAME)

    @property
    def source_dir(self) -> str:
        """Where the uploaded archive is kept, for provenance."""
        return os.path.join(self.root, SOURCE_DIRNAME)

    @property
    def meta_path(self) -> str:
        return os.path.join(self.root, SESSION_META)

    @property
    def owner(self) -> str:
        """The client this session belongs to.

        Empty for a session written before ownership existed; nothing can open
        such a session, so it simply ages out (see `open_session`).
        """
        return str(self.read_meta().get("owner") or "")

    def frame_path(self, index: int) -> str:
        """Where frame ``index`` would live, whether or not it exists yet."""
        if index < 0:
            raise InvalidRequest("A frame index cannot be negative.")
        return os.path.join(self.frames_dir, frame_name(index))

    def frame_names(self) -> List[str]:
        """The clip's frames, in playback order.

        A session built from a frame folder keeps the archive's own names, and
        the order the archive recorded is the authority — the names themselves
        only let us guess. Recorded names that are no longer on disk (a partial
        copy) are dropped, and an empty record falls back to the listing.
        """
        recorded = self.read_meta().get("frame_names")
        if isinstance(recorded, list) and recorded:
            present = set(list_frame_names(self.frames_dir))
            kept = [str(name) for name in recorded if str(name) in present]
            if kept:
                return kept
        return list_frame_names(self.frames_dir)

    def frame_name_at(self, index: int) -> str:
        """Name of frame ``index`` without listing the directory.

        Falls back to the canonical 8-digit name, which is what a session built
        from a video always uses.
        """
        if index < 0:
            raise InvalidRequest("A frame index cannot be negative.")
        recorded = self.read_meta().get("frame_names")
        if isinstance(recorded, list) and 0 <= index < len(recorded):
            return str(recorded[index])
        return frame_name(index)

    def resolve_frame(self, index: int) -> str:
        """Absolute path of frame ``index``, or ``NotFound`` when it is absent.

        Unlike ``frame_path`` (which names where a frame *would* live, using the
        canonical pattern), this resolves the name the session actually recorded.
        The name is reduced to its basename before it is joined to ``frames/``,
        so a name carried in from an archive can never reach outside the
        session's own frame directory — this is the only place that guard lives,
        so every reader shares it.
        """
        name = os.path.basename(self.frame_name_at(index))
        path = os.path.join(self.frames_dir, name)
        if not os.path.isfile(path):
            raise NotFound(f"Frame {index} is not part of session {self.id}.")
        return path

    def update_meta(self, **fields: Any) -> None:
        """Merge fields into the session's provenance file."""
        meta = self.read_meta()
        meta.update(fields)
        self.write_meta(meta)

    def frame_count(self) -> int:
        return len(self.frame_names())

    def touch(self) -> None:
        """Stamp the session as used now, so the sweeper leaves it alone."""
        now = time.time()
        with contextlib.suppress(OSError):
            os.utime(self.root, (now, now))

    def read_meta(self) -> Dict[str, Any]:
        try:
            with open(self.meta_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def write_meta(self, meta: Mapping[str, Any]) -> None:
        _write_json_atomic(self.meta_path, dict(meta))

    def size_bytes(self) -> int:
        return directory_size(self.root)

    def delete(self) -> None:
        """Remove the session and everything in it."""
        shutil.rmtree(self.root, ignore_errors=True)


def create_session(
    temp_dir: str,
    *,
    owner: str,
    meta: Optional[Mapping[str, Any]] = None,
) -> Session:
    """Make a fresh session directory owned by `owner`, and write its file.

    `owner` is keyword-only and required: a session nobody owns could not be
    opened by anyone, so there is no useful default to give it.
    """
    root = sessions_root(temp_dir)
    os.makedirs(root, exist_ok=True)
    session_id = uuid.uuid4().hex
    session = Session(id=session_id, root=os.path.join(root, session_id))
    os.makedirs(session.frames_dir, exist_ok=True)
    os.makedirs(session.source_dir, exist_ok=True)
    # The authoritative keys go last, so a caller cannot displace them by passing
    # `id` or `owner` in `meta`.
    session.write_meta({**(meta or {}), "id": session_id, "owner": owner})
    return session


def open_session(temp_dir: str, session_id: str, *, owner: str) -> Session:
    """Find a session owned by `owner`, or raise ``NotFound``.

    The id is checked against ``SESSION_ID_PATTERN`` before it is joined to a
    path, so a caller cannot escape ``temp_dir`` with ``..`` or a separator. A
    session belonging to someone else is reported as *unknown* rather than as
    forbidden: a non-owner must not learn that it exists.
    """
    if not SESSION_ID_PATTERN.match(session_id):
        raise NotFound(f"No such session: {session_id!r}.")
    root = os.path.join(sessions_root(temp_dir), session_id)
    if not os.path.isdir(root):
        raise NotFound(f"No such session: {session_id!r}.")
    session = Session(id=session_id, root=root)
    if session.owner != owner:
        raise NotFound(f"No such session: {session_id!r}.")
    return session


def delete_session(temp_dir: str, session_id: str, *, owner: str) -> bool:
    """Delete a session owned by `owner`; ``False`` when it was already gone."""
    try:
        session = open_session(temp_dir, session_id, owner=owner)
    except NotFound:
        return False
    session.delete()
    return True


def _owner_of(path: str) -> str:
    """The owner recorded for the session directory at `path`."""
    return Session(id=os.path.basename(path), root=path).owner


def sweep_sessions(
    temp_dir: str,
    ttl_seconds: int,
    *,
    max_bytes: Optional[int] = None,
    max_bytes_per_owner: Optional[int] = None,
    now: Optional[float] = None,
) -> List[str]:
    """Reap idle sessions, then over-quota ones, cheapest to lose first.

    Three passes, and the difference between them is what makes this safe for
    more than one client:

    1. **Idle** (per session) — the stamp is the session directory's mtime, which
       `Session.touch` refreshes on every use. Global, because idleness is each
       session's own business.
    2. **Per owner** — an owner over its own budget loses its own
       least-recently-used sessions. Without this pass one client uploading a
       large project could evict another client's *active* session, and that
       session's frames would start failing mid-review.
    3. **Global** (`max_bytes`) — the disk backstop, over every owner alike.
    """
    root = sessions_root(temp_dir)
    if not os.path.isdir(root):
        return []
    stamp = time.time() if now is None else now

    entries: List[tuple] = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        with contextlib.suppress(OSError):
            entries.append((os.path.getmtime(path), name, path))

    removed: List[str] = []
    survivors: List[tuple] = []
    for modified, name, path in sorted(entries):
        if stamp - modified > ttl_seconds:
            shutil.rmtree(path, ignore_errors=True)
            removed.append(name)
        else:
            survivors.append((modified, name, path))

    if max_bytes_per_owner is not None:
        by_owner: Dict[str, List[tuple]] = {}
        for entry in survivors:
            by_owner.setdefault(_owner_of(entry[2]), []).append(entry)
        survivors = []
        for owned in by_owner.values():
            total = sum(directory_size(path) for _m, _n, path in owned)
            for entry in sorted(owned):
                if total <= max_bytes_per_owner:
                    survivors.append(entry)
                    continue
                size = directory_size(entry[2])
                shutil.rmtree(entry[2], ignore_errors=True)
                removed.append(entry[1])
                total -= size

    if max_bytes is not None:
        total = sum(directory_size(path) for _m, _n, path in survivors)
        for modified, name, path in sorted(survivors):
            if total <= max_bytes:
                break
            size = directory_size(path)
            shutil.rmtree(path, ignore_errors=True)
            removed.append(name)
            total -= size

    return removed
