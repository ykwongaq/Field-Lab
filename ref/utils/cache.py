"""On-disk storage for keyframe probes and anchor decisions.

Masks are stored as COCO RLE (via ``pycocotools``), not as raw arrays: a single
1080x1920 boolean mask would be ~2 MB in JSON, whereas its RLE is a few KB.

Everything written here is written atomically (temp file + ``os.replace``) so a
killed worker cannot leave a half-written cache behind. This mirrors the
``--checkpoint_windows`` convention already used by ``1_sam3_inference.py``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, Optional

import numpy as np

from .geometry import as_bool_mask


def _mask_utils():
    """Import ``pycocotools.mask`` lazily (keeps module import GPU-free)."""
    from pycocotools import mask as mask_utils

    return mask_utils


def mask_to_json_rle(mask) -> Dict[str, Any]:
    """Encode a 2-D boolean mask as a JSON-serialisable COCO RLE dict."""
    arr = np.asfortranarray(as_bool_mask(mask).astype(np.uint8))
    rle = _mask_utils().encode(arr)
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return {"size": [int(rle["size"][0]), int(rle["size"][1])], "counts": counts}


def json_rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    """Decode a JSON COCO RLE dict (str ``counts``) back to a boolean mask."""
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = counts.encode("utf-8")
    decoded = _mask_utils().decode(
        {"size": [int(rle["size"][0]), int(rle["size"][1])], "counts": counts}
    )
    return as_bool_mask(decoded)


def anchor_cache_path(cache_root: str, rel_path: str, video_name: str) -> str:
    """Per-video cache file path under ``cache_root``."""
    parts = [cache_root]
    if rel_path and rel_path not in (".", ""):
        parts.append(rel_path)
    parts.append(f"{video_name}.json")
    return os.path.join(*parts)


def save_anchor_cache(path: str, payload: Dict[str, Any]) -> str:
    """Atomically write ``payload`` as JSON, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    os.replace(tmp_path, path)
    return path


def load_anchor_cache(path: str) -> Optional[Dict[str, Any]]:
    """Load a cache file, or ``None`` when it is missing/corrupt.

    A corrupt cache must never abort a run: the caller re-probes instead.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def append_jsonl(path: str, records: Iterable[Dict[str, Any]], lock=None) -> None:
    """Append records to a JSONL file, optionally under a multiprocessing lock.

    Workers each append their own lines, so the lock (when given) only has to
    cover the write itself.
    """
    records = [record for record in records if record is not None]
    if not records:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lines = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    if lock is not None:
        with lock:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(lines)
    else:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(lines)
