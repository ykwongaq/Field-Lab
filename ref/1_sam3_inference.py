"""Run SAM3 box+text prompts on the MammAlps-S2 videos, window by window.

Each video is split into overlapping sliding windows (``--window_size`` frames
with ``--overlap`` shared frames).  Every window is loaded into its own SAM3
session; for each tracked instance the species name is used as the text prompt
and the largest detection box (normalized xywh) inside the window is used as the
box prompt, then the prompt is propagated through the whole window.  Frames
that fall in two windows keep the mask from the window where they are most
interior, so the overlaps stitch the per-window results into one smooth
tracklet.

With ``--anchors`` the first/middle/last detection of every track inside the
window is additionally probed with the SAM3 image model, and the resulting
masks are injected into the tracker as authoritative keyframe anchors
(``add_mask``).  An anchor is only accepted when the probe is confident *and*
its mask agrees with that frame's ground-truth box, so a probe that latched
onto a different instance is dropped rather than written into memory.  Every
decision is appended to ``anchor_decisions.jsonl``; the reusable pieces live in
``utils/`` (see ``utils/planner.py``).

The result of every video is written as one JSON file that mirrors the regular
video-segmentation dataset layout
(``{info, videos, annotations, categories, video_np_pairs}``), i.e. the same
schema produced by ``1_split_annotation.py``:

* ``videos[0].file_names`` are frame paths relative to ``JPEGImages_30fps``.
* every ``annotations`` entry is one tracked object with per-frame COCO RLE
  ``segmentations`` (``None`` when the object is not visible), per-frame xywh
  ``bboxes`` / ``areas`` (``None`` when not visible), ``category_id`` and
  ``noun_phrase``.  As in the dataset, the last frame is not annotated, so the
  per-frame lists hold ``num_frames - 1`` entries.

Examples::

    python 1_sam3_inference.py                          # whole dataset, default GPUs
    python 1_sam3_inference.py --max_videos 3           # quick test
    python 1_sam3_inference.py --gpus 0,1,2             # one SAM3 worker per GPU
    python 1_sam3_inference.py --overlap 8 --checkpoint_windows
"""

import argparse
import datetime
import json
import multiprocessing
import os
import shutil
import sys
import time

import numpy as np
from PIL import Image
from pycocotools import mask as maskUtils
from sam3.model_builder import build_sam3_video_predictor

from utils import (
    AnchorConfig,
    KeyframeAnchorPlanner,
    anchor_cache_path,
    build_probe,
    outputs_to_mask_track,
    position_count,
    prompt_object_id,
    run_track,
)
from utils.keyframes import POSITION_PRESETS

# MammAlps-S2 species (as used in the detection files) -> global category id.
# The ids are stable across the whole dataset and match `2_visualize.py`.
SPECIES_TO_CATEGORY_ID = {
    "fox": 0,
    "marten": 1,
    "roe_deer": 2,
    "chamois": 3,
    "red_deer": 4,
    "hare": 5,
    "wolf": 6,
}
FALLBACK_CATEGORY_ID = 0
FALLBACK_SPECIES = "animal"

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def rle_to_json(rle):
    """Turn a pycocotools RLE dict into a JSON-serializable dict (str counts)."""
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return {
        "size": [int(rle["size"][0]), int(rle["size"][1])],
        "counts": counts,
    }


def _json_rle_to_pycoco(rle):
    """Turn a JSON RLE dict (str counts) back into a pycocotools RLE dict."""
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = counts.encode("utf-8")
    return {
        "size": [int(rle["size"][0]), int(rle["size"][1])],
        "counts": counts,
    }


def encode_mask(mask):
    """Encode a ``(H, W)`` bool ndarray into a JSON-serializable RLE dict."""
    return rle_to_json(maskUtils.encode(np.asfortranarray(mask.astype(np.uint8))))


def list_frame_files(frame_folder):
    """Return the sorted list of image file names in a video folder."""
    return sorted(
        name for name in os.listdir(frame_folder) if name.lower().endswith(_IMAGE_EXTS)
    )


def build_track_prompts(annotation, num_frames, width, height):
    """Group detections by ``track_id`` and build the (box, text) prompts.

    Returns a list of ``(track_id, species, boxes)`` where ``boxes`` is the
    sorted list of ``(frame_id, box_xywh_norm, box_xyxy)`` for every frame the
    track is detected on.  ``box_xywh_norm`` is ``[x, y, w, h]`` in normalized
    0~1 coordinates (the format SAM3 expects) and ``box_xyxy`` keeps the pixel
    box for selecting the best predicted object afterwards.
    """
    dets_by_track = {}
    for frame in annotation.get("frames") or []:
        for det in frame.get("detections") or []:
            track_id = det.get("track_id")
            if track_id is None:
                continue
            dets_by_track.setdefault(int(track_id), []).append(
                (frame.get("frame_id", 0), det)
            )

    prompts = []
    for track_id in sorted(dets_by_track):
        dets = dets_by_track[track_id]
        attrs = dets[0][1].get("attributes") or {}
        species = attrs.get("Species") or FALLBACK_SPECIES

        boxes = []
        for frame_id, det in dets:
            try:
                frame_id = int(frame_id)
            except (TypeError, ValueError):
                frame_id = 0
            if not (0 <= frame_id < num_frames):
                continue

            x1, y1, x2, y2 = det["bbox"]
            nx = max(0.0, min(1.0, x1 / width))
            ny = max(0.0, min(1.0, y1 / height))
            nx2 = max(0.0, min(1.0, x2 / width))
            ny2 = max(0.0, min(1.0, y2 / height))
            nw = nx2 - nx
            nh = ny2 - ny
            if nw <= 0.0 or nh <= 0.0:
                continue  # degenerate box; nothing to prompt

            boxes.append((frame_id, [nx, ny, nw, nh], [x1, y1, x2, y2]))

        if boxes:
            boxes.sort(key=lambda item: item[0])
            prompts.append((track_id, species, boxes))
    return prompts


def make_windows(num_frames, window_size, overlap):
    """Split ``[0, num_frames)`` into overlapping windows.

    Returns a list of ``(start, end)`` half-open frame ranges.  The stride is
    ``window_size - overlap`` (must be positive); the final window is truncated
    to ``num_frames`` and any window fully covered by its predecessor is
    dropped.
    """
    stride = window_size - overlap
    if stride <= 0:
        raise ValueError("window_size must be greater than overlap")
    windows = []
    prev_end = 0
    start = 0
    while start < num_frames:
        end = min(start + window_size, num_frames)
        if end > prev_end:
            windows.append((start, end))
            prev_end = end
        start += stride
    return windows


def rle_track_to_annotation(
    rle_track,
    num_frames,
    height,
    width,
    video_name,
    category_id,
    noun_phrase,
    ann_id,
):
    """Convert a ``{frame_idx: json_rle}`` dict into one dataset annotation."""
    segmentations = []
    bboxes = []
    areas = []
    for frame_idx in range(num_frames - 1):
        rle = rle_track.get(frame_idx)
        if rle is None:
            segmentations.append(None)
            bboxes.append(None)
            areas.append(None)
            continue
        pycoco_rle = _json_rle_to_pycoco(rle)
        segmentations.append(rle)
        x, y, w, h = maskUtils.toBbox(pycoco_rle)
        bboxes.append([float(x), float(y), float(w), float(h)])
        areas.append(float(maskUtils.area(pycoco_rle)))

    return {
        "id": ann_id,
        "segmentations": segmentations,
        "bboxes": bboxes,
        "areas": areas,
        "iscrowd": 0,
        "video_id": video_name,
        "height": height,
        "width": width,
        "category_id": category_id,
        "noun_phrase": noun_phrase,
    }


def window_checkpoint_dir(args, rel_path, video_name):
    """Per-video temp folder holding one checkpoint file per finished window."""
    return os.path.join(args.tmp_dataset, rel_path, video_name)


def load_window_checkpoints(tmp_dir, num_windows):
    """Return a list (one entry per window) of checkpointed track results."""
    completed = [None] * num_windows
    if not os.path.isdir(tmp_dir):
        return completed
    for idx in range(num_windows):
        path = os.path.join(tmp_dir, f"w_{idx:04d}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                completed[idx] = json.load(f)["tracks"]
    return completed


def save_window_checkpoint(tmp_dir, window_idx, window_range, window_results):
    """Atomically write one window's track results to the temp folder."""
    os.makedirs(tmp_dir, exist_ok=True)
    payload = {
        "window": [int(window_range[0]), int(window_range[1])],
        "tracks": window_results,
    }
    path = os.path.join(tmp_dir, f"w_{window_idx:04d}.json")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp_path, path)


def write_video_output(
    args,
    rel_path,
    video_name,
    video_id,
    frame_files,
    height,
    width,
    annotations,
    referenced,
):
    """Assemble and atomically write the per-video dataset JSON."""
    out_path = os.path.join(args.output_dataset, rel_path, f"{video_name}.json")

    counts = {}
    for ann in annotations:
        key = (ann["category_id"], ann["noun_phrase"])
        counts[key] = counts.get(key, 0) + 1
    video_np_pairs = [
        {
            "id": pair_id,
            "video_id": video_name,
            "category_id": category_id,
            "noun_phrase": species,
            "num_masklets": counts.get((category_id, species), 0),
        }
        for pair_id, (species, category_id) in enumerate(
            sorted(referenced.items(), key=lambda kv: kv[1]), start=1
        )
    ]

    categories = [
        {"id": category_id, "name": name}
        for name, category_id in sorted(referenced.items(), key=lambda kv: kv[1])
    ]

    output = {
        "info": {
            "version": "sam3-box-text",
            "date": datetime.date.today().isoformat(),
            "description": "SAM3 box+text inference on MammAlps-S2",
        },
        "videos": [
            {
                "id": video_id,
                "video_name": video_name,
                "file_names": [
                    f"{rel_path}/{video_name}/{frame_file}"
                    for frame_file in frame_files
                ],
                "height": height,
                "width": width,
                "length": len(frame_files),
                "location_id": video_name,
            }
        ],
        "annotations": annotations,
        "categories": categories,
        "video_np_pairs": video_np_pairs,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, out_path)
    return out_path


def process_video(args, predictor, rel_path, json_filename, video_id, probe=None):
    """Run windowed SAM3 inference for one video (sliding ``--window_size`` frames)."""
    video_name = os.path.splitext(json_filename)[0]
    annotation_file = os.path.join(args.annotation_dataset, rel_path, json_filename)
    frame_folder = os.path.join(args.frame_dataset, rel_path, video_name)
    out_path = os.path.join(args.output_dataset, rel_path, f"{video_name}.json")

    if args.skip_existing and os.path.exists(out_path):
        return "skip", out_path, "already exists"

    with open(annotation_file, "r", encoding="utf-8") as f:
        annotation = json.load(f)

    info = annotation["info"]
    height = int(info["height"])
    width = int(info["width"])

    frame_files = list_frame_files(frame_folder)
    num_frames = len(frame_files)
    if num_frames == 0:
        return "fail", out_path, "no frames found"

    prompts = build_track_prompts(annotation, num_frames, width, height)
    track_species = {track_id: species for track_id, species, _boxes in prompts}
    track_category = {
        track_id: SPECIES_TO_CATEGORY_ID.get(species, FALLBACK_CATEGORY_ID)
        for track_id, species, _boxes in prompts
    }
    referenced = {}
    for track_id, species, _boxes in prompts:
        referenced.setdefault(species, track_category[track_id])

    windows = make_windows(num_frames, args.window_size, args.overlap)

    # Optional keyframe anchoring. The probes need the window's frames, so the
    # loader reads from the images already loaded for the current window (the
    # session keeps its own copies anyway).
    window_images = {}
    planner = None
    if args.anchors:

        def load_window_frame(global_frame):
            image = window_images.get(int(global_frame))
            if image is None:
                image = Image.open(
                    os.path.join(frame_folder, frame_files[int(global_frame)])
                ).convert("RGB")
            return image

        planner = KeyframeAnchorPlanner(
            probe=probe,
            config=args.anchor_config,
            width=width,
            height=height,
            frame_loader=load_window_frame,
            keyframe_count=args.keyframe_count,
            cache_path=(
                anchor_cache_path(args.anchor_cache, rel_path, video_name)
                if args.anchor_cache
                else None
            ),
            decisions_path=args.anchor_log,
            log_lock=getattr(args, "anchor_log_lock", None),
            video_label=f"{rel_path}/{video_name}",
        )

    # track_id -> {global_frame: (interior_score, json_rle)}
    track_accum = {}

    tmp_dir = None
    completed = [None] * len(windows)
    if args.checkpoint_windows:
        tmp_dir = window_checkpoint_dir(args, rel_path, video_name)
        completed = load_window_checkpoints(tmp_dir, len(windows))

    def merge_rle(track_id, global_frame, rle, w_start, w_end):
        # Frames that fall in two windows keep the mask from the window where
        # they are most interior, which gives a smooth overlap hand-off.
        score = min(global_frame - w_start, w_end - 1 - global_frame)
        acc = track_accum.setdefault(track_id, {})
        cur = acc.get(global_frame)
        if cur is None or score > cur[0]:
            acc[global_frame] = (score, rle)

    for w_idx, (w_start, w_end) in enumerate(windows):
        if completed[w_idx] is not None:
            for entry in completed[w_idx]:
                for gf_str, rle in entry["frames"].items():
                    merge_rle(entry["track_id"], int(gf_str), rle, w_start, w_end)
            continue

        window_len = w_end - w_start
        pil_frames = [
            Image.open(os.path.join(frame_folder, frame_files[i])).convert("RGB")
            for i in range(w_start, w_end)
        ]
        response = predictor.handle_request(
            request=dict(type="start_session", resource_path=pil_frames)
        )
        session_id = response["session_id"]
        if planner is not None:
            window_images.clear()
            window_images.update(
                {w_start + offset: image for offset, image in enumerate(pil_frames)}
            )
        del pil_frames
        window_results = []
        try:
            for track_id, species, boxes in prompts:
                wboxes = [b for b in boxes if w_start <= b[0] < w_end]
                if not wboxes:
                    continue
                # The largest detection box in the window anchors the prompt.
                prompt_frame, box_norm, box_abs = max(
                    wboxes, key=lambda b: (b[2][2] - b[2][0]) * (b[2][3] - b[2][1])
                )
                local_frame = prompt_frame - w_start
                anchor_masks = None
                if planner is not None:
                    # Skip an anchor on the seed frame: the box prompt already
                    # conditions that frame, so a mask there adds nothing.
                    anchor_masks = [
                        (anchor.frame_index - w_start, anchor.mask)
                        for anchor in planner.anchors_for_track(
                            track_id, species, wboxes, w_start, w_end
                        )
                        if anchor.frame_index != prompt_frame
                    ] or None
                outputs_per_frame = run_track(
                    predictor,
                    session_id,
                    local_frame,
                    species,
                    box_norm,
                    prompt_box_xyxy=box_abs,
                    anchor_masks=anchor_masks,
                )
                mask_track = outputs_to_mask_track(
                    outputs_per_frame, window_len, local_frame, box_abs
                )
                if not mask_track:
                    continue
                frames_rle = {}
                for local_idx, mask in mask_track.items():
                    global_frame = w_start + local_idx
                    rle = encode_mask(mask)
                    frames_rle[str(global_frame)] = rle
                    merge_rle(track_id, global_frame, rle, w_start, w_end)
                window_results.append(
                    {
                        "track_id": track_id,
                        "species": species,
                        "category_id": track_category[track_id],
                        "frames": frames_rle,
                    }
                )
        finally:
            try:
                predictor.handle_request(
                    request=dict(type="close_session", session_id=session_id)
                )
            except Exception:
                pass

        if args.checkpoint_windows:
            save_window_checkpoint(tmp_dir, w_idx, (w_start, w_end), window_results)

    if planner is not None:
        planner.flush()
        window_images.clear()

    annotations = []
    ann_id = 0
    for track_id in sorted(track_accum):
        rle_track = {gf: rle for gf, (_score, rle) in track_accum[track_id].items()}
        annotations.append(
            rle_track_to_annotation(
                rle_track,
                num_frames,
                height,
                width,
                video_name,
                track_category.get(track_id, FALLBACK_CATEGORY_ID),
                track_species.get(track_id, FALLBACK_SPECIES),
                ann_id,
            )
        )
        ann_id += 1

    write_video_output(
        args,
        rel_path,
        video_name,
        video_id,
        frame_files,
        height,
        width,
        annotations,
        referenced,
    )

    if args.checkpoint_windows and tmp_dir:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    n_obj = len(annotations)
    anchor_note = ""
    if planner is not None:
        stats = planner.stats
        considered = stats["accepted"] + stats["rejected"]
        anchor_note = (
            f", anchors {stats['accepted']}/{considered} "
            f"(probes {stats['probes']}, cached {stats['cache_hits']}, "
            f"errors {stats['errors']})"
        )
    return "ok", out_path, f"{n_obj} object(s), {len(windows)} window(s){anchor_note}"


def _is_oom(exc):
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _free_gpu_cache():
    try:
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


def _log_failure(args, rel_path, json_filename, kind, message, lock=None):
    """Append one failure record to ``failed_videos.jsonl`` in the output dir."""
    try:
        out_dir = args.output_dataset
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "failed_videos.jsonl")
        entry = {
            "rel_path": rel_path,
            "video": json_filename,
            "kind": kind,
            "message": message,
            "time": datetime.datetime.now().isoformat(),
        }
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        if lock is not None:
            with lock:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line)
        else:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception:
        pass


def process_task(
    args,
    predictor,
    tag,
    rel_path,
    json_filename,
    video_id,
    done,
    total,
    stats,
    fail_lock=None,
    probe=None,
):
    """Run one video and update the worker's ``stats`` / progress line.

    Any per-video error (OOM or otherwise) is caught here so a single bad video
    can never kill the worker, and every failure is appended to
    ``failed_videos.jsonl`` for later re-runs.
    """
    video_name = os.path.splitext(json_filename)[0]
    t0 = time.time()
    try:
        status, out_path, message = process_video(
            args, predictor, rel_path, json_filename, video_id, probe=probe
        )
    except RuntimeError as exc:
        if _is_oom(exc):
            _free_gpu_cache()
            stats["oom"] = stats.get("oom", 0) + 1
            _log_failure(args, rel_path, json_filename, "oom", str(exc), fail_lock)
            print(f"{tag} !! OOM on {video_name}: {exc}", flush=True)
            return
        _log_failure(args, rel_path, json_filename, "error", str(exc), fail_lock)
        raise
    except Exception as exc:
        stats["fail"] = stats.get("fail", 0) + 1
        _log_failure(
            args,
            rel_path,
            json_filename,
            "error",
            f"{type(exc).__name__}: {exc}",
            fail_lock,
        )
        print(f"{tag} !! FAIL {video_name}: {type(exc).__name__}: {exc}", flush=True)
        return
    stats[status] = stats.get(status, 0) + 1
    print(
        f"{tag} [{done}/{total}] {video_name}: {status} ({message}) "
        f"in {time.time() - t0:.1f}s",
        flush=True,
    )


def build_worker_probe(args, predictor):
    """Build the keyframe probe for a worker (``None`` when anchors are off).

    The image backend loads a second model onto the GPU this worker is pinned
    to, so it is only built when ``--anchors`` actually asks for it.
    """
    if not args.anchors:
        return None
    if args.anchor_mask_source == "video":
        return build_probe("video", predictor=predictor)
    return build_probe(
        "image",
        checkpoint_path=args.probe_checkpoint or args.checkpoint,
        device="cuda",
        resolution=args.probe_resolution,
    )


def run_shard(args, gpu_id, tasks):
    """Build one SAM3 model on ``gpu_id`` and process ``tasks`` in-process.

    Used for the single-GPU path (no spawn overhead).
    """
    predictor = build_sam3_video_predictor(
        checkpoint_path=args.checkpoint, gpus_to_use=[gpu_id]
    )
    probe = build_worker_probe(args, predictor)
    tag = f"[gpu{gpu_id}]"
    stats = {"ok": 0, "skip": 0, "fail": 0, "oom": 0}

    try:
        total = len(tasks)
        for done, (rel_path, json_filename, video_id) in enumerate(tasks, start=1):
            process_task(
                args,
                predictor,
                tag,
                rel_path,
                json_filename,
                video_id,
                done,
                total,
                stats,
                probe=probe,
            )
    finally:
        if probe is not None:
            probe.close()
        try:
            predictor.shutdown()
        except Exception:
            pass

    print(
        f"{tag} done: {stats['ok']} ok, {stats['skip']} skipped, "
        f"{stats['fail']} failed, {stats['oom']} OOM.",
        flush=True,
    )
    return stats


def run_worker(args, gpu_id, task_queue, total, counter, lock):
    """Queue-draining SAM3 worker pinned to one GPU.

    Builds its own SAM3 model, then pulls videos from the shared queue until a
    ``None`` sentinel arrives.  Workers grab the next video as soon as they are
    idle, so longer videos do not stall other GPUs.
    """
    predictor = build_sam3_video_predictor(
        checkpoint_path=args.checkpoint, gpus_to_use=[gpu_id]
    )
    probe = build_worker_probe(args, predictor)
    # Each worker already holds a cross-process lock for the failure log; reuse
    # it for the anchor decision log so the two writes cannot interleave.
    args.anchor_log_lock = lock
    tag = f"[gpu{gpu_id}]"
    stats = {"ok": 0, "skip": 0, "fail": 0, "oom": 0}

    try:
        while True:
            task = task_queue.get()
            if task is None:
                break
            rel_path, json_filename, video_id = task
            with lock:
                counter.value += 1
                done = counter.value
            process_task(
                args,
                predictor,
                tag,
                rel_path,
                json_filename,
                video_id,
                done,
                total,
                stats,
                fail_lock=lock,
                probe=probe,
            )
    finally:
        if probe is not None:
            probe.close()
        try:
            predictor.shutdown()
        except Exception:
            pass

    print(
        f"{tag} done: {stats['ok']} ok, {stats['skip']} skipped, "
        f"{stats['fail']} failed, {stats['oom']} OOM.",
        flush=True,
    )


def main(args):
    args.keyframe_count = position_count(args.keyframe_positions)
    args.anchor_config = AnchorConfig(
        min_score=args.anchor_min_score,
        min_containment=args.anchor_min_containment,
        max_anchors=args.anchor_max_per_track,
    )
    args.anchor_log_lock = None
    if args.anchors:
        if args.anchor_log is None:
            args.anchor_log = os.path.join(
                args.output_dataset, "anchor_decisions.jsonl"
            )
        if args.anchor_cache is None:
            args.anchor_cache = args.output_dataset + "_anchors"
    else:
        # Anchoring is off: no planner is created, so no cache or log either.
        args.anchor_log = None
        args.anchor_cache = None

    if not os.path.exists(args.annotation_dataset):
        print(f"Annotation dataset not found: {args.annotation_dataset}")
        return

    # Stable, global video ordering (also used as videos[].id).
    all_videos = []
    for root, _dirs, files in os.walk(args.annotation_dataset):
        rel_path = os.path.relpath(root, args.annotation_dataset)
        for filename in sorted(files):
            if filename.endswith(".json"):
                all_videos.append((rel_path, filename))
    all_videos.sort(key=lambda item: (item[0], item[1]))
    if args.max_videos is not None:
        all_videos = all_videos[: args.max_videos]

    tasks = [
        (rel_path, filename, video_id)
        for video_id, (rel_path, filename) in enumerate(all_videos)
    ]

    print(f"Running SAM3 box+text inference on {len(tasks)} video(s).")

    if len(args.gpus) == 1:
        run_shard(args, args.gpus[0], tasks)
        return

    # One persistent SAM3 worker per GPU.  Workers pull the next video from a
    # shared queue, so a GPU that finishes a short video immediately picks up
    # another one instead of idling behind a static round-robin split.
    ctx = multiprocessing.get_context("spawn")
    task_queue = ctx.Queue()
    for task in tasks:
        task_queue.put(task)
    for _ in args.gpus:
        task_queue.put(None)  # one sentinel per worker

    counter = ctx.Value("i", 0)
    lock = ctx.Lock()
    processes = []
    for gpu_id in args.gpus:
        proc = ctx.Process(
            target=run_worker,
            args=(args, gpu_id, task_queue, len(tasks), counter, lock),
        )
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join()

    failed = [proc.exitcode for proc in processes if proc.exitcode != 0]
    if failed:
        print(f"WARNING: {len(failed)}/{len(processes)} worker(s) failed: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frame_dataset",
        type=str,
        default="/mnt/hdd2/davidwong/data/VideoSegmentation/public_datasets/MammAlps-S2/JPEGImages_30fps",
    )
    parser.add_argument(
        "--annotation_dataset",
        type=str,
        default="/mnt/hdd2/davidwong/data/VideoSegmentation/public_datasets/MammAlps-S2/annotations",
    )
    parser.add_argument(
        "--output_dataset",
        type=str,
        default="/mnt/hdd2/davidwong/data/VideoSegmentation/public_datasets/MammAlps-S2/segmentation",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/mnt/hdd2/davidwong/models/sam3/sam3.pt",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="2,3",
        help="Comma-separated GPU ids, e.g. '0,1,2'. Default: '1,2,3,4'.",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Only process the first N videos (for quick testing).",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=64,
        help="Size of each sliding window in frames. Default: 64.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=6,
        help="Number of frames shared between consecutive windows. Default: 6.",
    )
    parser.add_argument(
        "--checkpoint_windows",
        action="store_true",
        help=(
            "Save each finished window to a temp folder (under "
            "<output_dataset>_tmp) and resume from there on re-runs."
        ),
    )
    parser.add_argument(
        "--anchors",
        action="store_true",
        help=(
            "Probe the first/middle/last detection of every track in each "
            "window and inject the accepted masks as keyframe anchors."
        ),
    )
    parser.add_argument(
        "--keyframe_positions",
        type=str,
        default="first_mid_last",
        choices=sorted(POSITION_PRESETS),
        help="Keyframes to probe per track per window. Default: first_mid_last.",
    )
    parser.add_argument(
        "--anchor_min_score",
        type=float,
        default=0.5,
        help="Minimum probe confidence for an anchor. Default: 0.5.",
    )
    parser.add_argument(
        "--anchor_min_containment",
        type=float,
        default=0.9,
        help=(
            "Minimum |probe mask AND ground-truth box| / |probe mask|; guards "
            "against anchoring a different instance. Default: 0.9."
        ),
    )
    parser.add_argument(
        "--anchor_max_per_track",
        type=int,
        default=3,
        help="Maximum accepted anchors per track per window. Default: 3.",
    )
    parser.add_argument(
        "--anchor_mask_source",
        type=str,
        default="image",
        choices=("image", "video"),
        help=(
            "Probe backend: the SAM3 image model, or a one-frame session on "
            "the video model. Default: image."
        ),
    )
    parser.add_argument(
        "--anchor_cache",
        type=str,
        default=None,
        help=(
            "Folder for per-video probe caches; re-runs reuse them instead of "
            "re-probing. Default: <output_dataset>_anchors when --anchors is set."
        ),
    )
    parser.add_argument(
        "--anchor_log",
        type=str,
        default=None,
        help=(
            "JSONL file recording every anchor accept/reject decision. "
            "Default: <output_dataset>/anchor_decisions.jsonl."
        ),
    )
    parser.add_argument(
        "--probe_checkpoint",
        type=str,
        default=None,
        help="Checkpoint for the image probe. Default: --checkpoint.",
    )
    parser.add_argument(
        "--probe_resolution",
        type=int,
        default=1008,
        help="Input resolution of the image probe. Default: 1008.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip videos whose output file already exists.",
    )

    args = parser.parse_args()
    args.gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    args.gpus = list(dict.fromkeys(args.gpus))  # drop duplicates, keep order
    if not args.gpus:
        parser.error("--gpus must contain at least one GPU id")
    if args.overlap >= args.window_size:
        parser.error("--overlap must be smaller than --window_size")
    args.tmp_dataset = args.output_dataset + "_tmp"
    main(args)
