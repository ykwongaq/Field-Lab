# Implementation Plan — Mask-Only Prompting + Windowed Propagation

> **Status:** awaiting sign-off · **Created:** 2026-09-30
> **Companion:** `docs/MASK_INTERACTION_SPEC.md` (interaction rules; this file is the build plan).
> **Reference implementation:** `ref/1_sam3_inference.py` + `ref/utils/` (windowed SAM 3 propagation).

---

## 0. Summary

Three structural changes:

1. **Inference moves to the official `sam3` package** (the checkout we read), replacing the
   Hugging Face `transformers` path. This is what makes mask-only prompting possible: the HF tracker
   exposes `add_inputs_to_inference_session(... input_masks=...)` but not the authoritative
   `add_mask` + `force_tracker_propagation` semantics, the per-frame `add_mask` re-anchoring, or the
   `Sam3Image.predict_inst` point prompt. Every mechanism in the spec's refinement loop (§8) exists
   only in the official package.
2. **Propagation is plan-driven and windowed**: one job per object, the requested range is split into
   overlapping windows, each window is its own SAM 3 session, and the next window is re-anchored with
   the previous window's masks. This removes the current hard `PROPAGATE_MAX_FRAMES=120` ceiling and
   the GPU-OOM risk that motivated it.
3. **Frames come from the server-side session** (`session_id` + frame indices), not from frames
   uploaded with every request. Kills the per-request upload, makes propagation to the end of a clip
   practical, and makes prompt latency independent of frame size.

Everything else follows: the image tools produce an RLE mask (point / box / text / polygon / brush),
and **the only thing the propagator accepts is a mask** (P4 in the spec).

---

## 1. Decisions to confirm before I start

| #      | Decision                                                                                                                         | Recommendation | Why it matters                                                                                                                                                                                |
| ------ | -------------------------------------------------------------------------------------------------------------------------------- | -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **C1** | Switch to the official `sam3` package (drop `transformers`)                                                                      | **Yes**        | Mask-only prompts, anchors and point prompts are official-repo only (see §0.1). Blocker: the package is **not installed in any environment on this machine**, and the checkpoint is HF-gated. |
| **C2** | "Mask prompt only" applies to **propagation**; the image endpoint still takes point / box / text (that is how a mask is created) | **Yes**        | Confirms the read of the request. If you meant the image model should also drop text/box, the tool list shrinks and `Sam3Processor` is not needed at all.                                     |
| **C3** | Propagation reads frames from `session_id`; the anchor mask still travels in the request body                                    | **Yes**        | The annotation lives in the browser-side archive, so the mask must come from the client; the pixels need not.                                                                                 |
| **C4** | Jobs are owned by the backend (FIFO queue, one GPU job at a time); the frontend displays the queue                               | **Yes**        | Enforces the one-object-per-job rule server-side, so a multi-select cannot accidentally widen a request.                                                                                      |
| **C5** | Two model instances may be resident (image + video). Default `sam3_max_resident_models=1` evicts on switch                       | **Yes**        | ~2 × (848M params, bf16) + activations on one GPU is tight; eviction keeps a single-GPU box working. Set to 2 if you have headroom, or pin each to its own device.                            |
| **C6** | `enable_inst_interactivity=True` for the image model (required for point prompts)                                                | **Yes**        | Loads the tracker weights into the image model; slightly larger footprint. Without it there is no point prompt.                                                                               |

---

## 2. Backend

### 2.1 New modules

| File                          | Responsibility                                                                                                                                                                                                                                                                                                                                                                             |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `src/domain/windows.py`       | **Pure** (no torch): `plan_windows(anchor, first, last, window, overlap, direction)` → ordered `Window(start, end, anchors, travel)` list; `merge_series_score(...)` interior-score stitching; frame-range validation. Unit-testable without a GPU.                                                                                                                                        |
| `src/domain/mask_prompt.py`   | Prompt value objects: `PointPrompt`, `BoxPrompt`, `TextPrompt` → validation + normalisation (pixels → `cxcywh`/`xywh` in 0..1), and `union_rle` / `instances` helpers. Replaces the exemplar-box logic.                                                                                                                                                                                    |
| `src/inference/models.py`     | `ModelManager`: lazy `image_model()` / `video_predictor()`, LRU eviction, device resolution, `shutdown()`, `status()`. Single place that imports `sam3`.                                                                                                                                                                                                                                   |
| `src/inference/sam3_image.py` | `Sam3ImageService`: `segment(session, frame_index, prompt)` → `{mask_rle, score, instances[], bbox, area, timings}`. Routes point → `predict_inst`, box → `add_geometric_prompt`, text → `set_text_prompt`. Keeps an LRU of `set_image` states keyed by `(session_id, frame_index)`.                                                                                                       |
| `src/inference/sam3_video.py` | `Sam3VideoPropagator`: `propagate(session, anchor, mask, direction, first, last, progress_cb, cancel)` → `{frame: rle}`. Owns the window loop: per window `start_session(frames)` → `add_mask` anchors → `propagate_in_video(force_tracker_propagation=True)` → `close_session`; stitches with `merge_series_score`. Takes the predictor as a constructor argument so tests inject a stub. |
| `src/core/jobs.py`            | `PropagationJob` + `JobRegistry`: FIFO queue, one worker thread, per-frame progress, cancellation `Event`, incremental result reads, TTL sweep, shutdown.                                                                                                                                                                                                                                  |

### 2.2 Changed modules

| File                          | Change                                                                                                                                                                                                                                                                                                                                                                                           |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `src/core/config.py`          | Drop `sam3_model` (HF id), `sam3_exemplar_fraction`, `sam3_tracker_model`. Add `sam3_image_checkpoint`, `sam3_bpe_path`, `sam3_device`, `sam3_max_resident_models`, `propagate_window_frames` (default 48), `propagate_overlap` (default 8), `propagate_anchor_max` (default 3), `propagate_job_ttl_seconds`. `propagate_max_frames` becomes the **window** size ceiling, not a request ceiling. |
| `src/api/deps.py`             | `Sam3Dep` → image service + job registry deps; `EnableSam3Dep` unchanged.                                                                                                                                                                                                                                                                                                                        |
| `src/api/serializers.py`      | `to_rle` / `to_foreground_runs` kept; add `job_response`, `segment_response` (instances), drop exemplar serialisation.                                                                                                                                                                                                                                                                           |
| `src/schemas/sam.py`          | `Sam3Status` unchanged in shape; segment request becomes a JSON body (no multipart), response gains `instances[]` and `candidates[]`.                                                                                                                                                                                                                                                            |
| `src/schemas/propagate.py`    | Job request/response models replacing the multipart window request.                                                                                                                                                                                                                                                                                                                              |
| `src/api/routes/propagate.py` | `POST /jobs`, `GET /jobs`, `GET /jobs/{id}?since=`, `DELETE /jobs/{id}`, `GET /status`.                                                                                                                                                                                                                                                                                                          |
| `src/api/routes/sam3.py`      | `GET /status`, `POST /segment` (JSON body, session-based).                                                                                                                                                                                                                                                                                                                                       |
| `src/inference/registry.py`   | Cache `ModelManager` + `JobRegistry`; `reset_services()` also shuts models down.                                                                                                                                                                                                                                                                                                                 |
| `src/core/lifespan.py`        | Cancel running jobs and shut the models down on exit; sweep finished jobs with the session sweeper.                                                                                                                                                                                                                                                                                              |
| `requirements.txt`            | Remove `transformers`; pin `torch>=2.7`, `pycocotools`, `pydantic`, `fastapi`, `pillow`; document the `pip install -e <sam3 checkout>` step.                                                                                                                                                                                                                                                     |

### 2.3 Deleted

| File                         | Reason                                                                                                                  |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `src/inference/sam3.py`      | HF `Sam3Model` + exemplar-box prompting (superseded).                                                                   |
| `src/inference/propagate.py` | HF tracker + per-request frame upload (superseded).                                                                     |
| `src/api/routes/masks.py`    | `/api/decode/masks`; `frontend/src/lib/rle.ts` already decodes pycocotools RLE in the browser.                          |
| `src/schemas/masks.py`       | Only used by the route above.                                                                                           |
| `src/domain/segmentation.py` | `exemplar_boxes_from_points`, `ExemplarBox`, `ConceptResult` — replaced by `mask_prompt.py`. `PromptPoint` moves there. |
| `domain/rle.py`              | Kept — still used for encoding/decoding on the backend.                                                                 |

### 2.4 API surface: old → new

| Old                                                                                            | New                                                                                                                                     |
| ---------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `POST /api/sam3/segment` (multipart: image + points + text + image_key)                        | `POST /api/sam3/segment` (JSON: `session_id`, `frame_index`, `prompt`)                                                                  |
| `POST /api/propagate` (multipart: frames[] + frame_indices + anchor + mask + backward/forward) | `POST /api/propagate/jobs` → `{job_id}` (JSON: `session_id`, `anchor_frame`, `mask`, `direction`, `first`, `last`)                      |
| —                                                                                              | `GET /api/propagate/jobs` (queue), `GET /api/propagate/jobs/{id}?since=` (progress + masks), `DELETE /api/propagate/jobs/{id}` (cancel) |
| `GET /api/propagate/status`                                                                    | Kept, reports the tracker model + window/overlap settings                                                                               |
| `POST /api/decode/masks`                                                                       | **Removed**                                                                                                                             |

---

## 3. The propagation algorithm

Direct adaptation of `ref/1_sam3_inference.py` + `ref/utils/tracking.py` to a single mask prompt.

**Input:** `session_id`, `anchor_frame` (_a_), anchor mask, `direction ∈ {forward, backward, both}`,
`first`, `last`.

**Planning** (`domain/windows.py`, mirrors the ref's `make_windows`):

```
stride = window - overlap            (window=48, overlap=8 → stride 40)
forward chain:  [a, min(a+window, last)] , then +stride, … until last
backward chain: [max(a-window+1, first), a] , then -stride, … until first
direction both → forward chain, then backward chain   (upstream's order)
```

The first window of a chain carries the user's mask as its anchor; every later window carries the
masks the previous window produced on the **overlap frames** (capped at `propagate_anchor_max`).

**Per window** (the request sequence from `ref/utils/tracking.py::run_track`, minus the prompt):

```python
session_id = predictor.handle_request(type="start_session", resource_path=<PIL frames of the window>)
predictor.handle_request(type="add_mask", session_id=…, frame_index=<local>, obj_id=1, mask=<bool HxW>)
for r in predictor.handle_stream_request(type="propagate_in_video", session_id=…,
                                         propagation_direction=…,
                                         force_tracker_propagation=True):
    …
predictor.handle_request(type="close_session", session_id=…)
```

Notes that matter:

- `resource_path` accepts an in-memory list of PIL frames (the ref relies on this), so a window never
  needs temp files on disk.
- `force_tracker_propagation=True` is required: `add_mask` records a _refine_ action, and without the
  flag the action history can resolve to `propagation_fetch` and return cached predictions instead of
  running the tracker.
- `obj_id` is `1` in every window — sessions are independent, and the frontend owns tracklet identity.
  `add_tracker_new_mask` registers a new object when the id is unknown, which is what makes mask-only
  prompting work with no text/box prompt.
- Masks are written back at **global** frame indices (`w_start + local`).

**Stitching:** a frame covered by two windows keeps the value from the window where it is most
interior (`score = min(f - start, end-1 - f)`), so the hand-off is deterministic rather than
last-writer-wins.

**Why windows solve OOM:** peak GPU memory is a function of one window's frame count, not of the
clip length, and the old blanket 120-frame ceiling disappears — a 5,000-frame "to the end" job is now
~125 windows of 48 frames, each with its own session.

**Progress and cancellation:** the job reports `{state, window_index, windows_total, frames_done,
frames_total}`; cancellation is checked between windows and between streamed frames, and the masks
produced so far are kept.

---

## 4. Model management

- One `ModelManager` per process; `Sam3ImageService` and `Sam3VideoPropagator` borrow from it.
- Image model: `build_sam3_image_model(checkpoint_path=…, bpe_path=…, enable_inst_interactivity=True)`.
- Video: `build_sam3_video_predictor(checkpoint_path=…, bpe_path=…)`. Its `shutdown()` exists and is
  called on eviction (it also releases the bf16 autocast context).
- Eviction is LRU with `sam3_max_resident_models` (default 1) + `torch.cuda.empty_cache()`; the
  status endpoint reports which models are loaded.
- The image service caches at most 2 `set_image` states so click-by-click prompting re-runs only the
  decoder.

---

## 5. Frontend

### 5.1 New / changed

| File                                               | Change                                                                                                                                                 |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `src/lib/sam3Api.ts` (new)                         | `fetchSam3Status`, `segment({sessionId, frameIndex, prompt})`. Replaces `samApi.ts`.                                                                   |
| `src/lib/propagateApi.ts` (new)                    | `fetchPropagateStatus`, `startPropagation`, `getJob(id, since)`, `cancelJob`, `listJobs`.                                                              |
| `src/lib/maskCache.ts` (renamed from `maskApi.ts`) | Same `MaskCache.resolveBatch` interface, but decodes **locally** with `lib/rle.ts` instead of `POST /api/decode/masks`. `VideoPanel.tsx` is unchanged. |
| `src/components/Workspace.tsx`                     | Per-frame drafts, staleness, job panel, new propagate controls, queue.                                                                                 |
| `src/components/Toolbar.tsx`                       | Tools: Review / Add mask / Edit mask / Propagate; method chips per tool (§3 of the spec).                                                              |
| `src/components/PropagationPanel.tsx` (new)        | Direction, range (`N frames` / `To end`), write policy, pre-run counts, run/cancel, queue list.                                                        |
| `src/components/TimelineStrip.tsx` (new)           | Per-frame state (accepted / draft / stale / lost / job progress).                                                                                      |

### 5.2 State changes in `Workspace.tsx`

| Today                                      | New                                                                            |
| ------------------------------------------ | ------------------------------------------------------------------------------ |
| `draft: RawRle \| null`                    | `drafts: Map<string, RawRle>` keyed `"${objectId}:${frame}"` (spec §6.2)       |
| `method: "sam" \| "polygon" \| "brush"`    | `method: "point" \| "box" \| "text" \| "polygon" \| "brush" \| "erase"`        |
| `prompt: PromptPoint[]` (→ exemplar boxes) | `prompt: {kind, points, box, text}` (→ one mask)                               |
| `propBack` / `propForward` numbers         | `direction`, `rangeMode ("frames" \| "toEnd")`, `rangeCount`, `writePolicy`    |
| `propRun` (single blocking run)            | `job: PropagationJob` (polled), `queue: PropagationJob[]`                      |
| —                                          | `stale: Map<string, number>` (frame → the corrected frame that invalidated it) |

### 5.3 Flows

- **Create:** click / box / text → `POST /api/sam3/segment` → draft. Polygon and brush compose into the
  same draft (`composeRle`, already present).
- **Propagate:** requires a draft or a committed mask on the current frame; the panel shows the counts
  before running; the job streams masks in, each becoming a _draft_ until accepted.
- **Refine:** correcting frame `k` marks that object's frames `> k` stale (forward) and enables
  **Re-propagate from here** (= the same job API with `first = k`).
- **Accept:** unchanged (`clip.replaceMask`), but now also clears the draft and the stale marks it
  replaces.

### 5.4 Deleted

- `src/lib/samApi.ts` (SAM 2 status, `segmentFrame`, `PropagateBackend`) — `sam2` no longer exists.
- `segmentConcept` / `Sam3SegmentResult.exemplars` (the exemplar-box model).
- The `semantic ? SAM 3 : SAM 2` branching in `Workspace.tsx`; the model is always SAM 3.
- `POST /api/decode/masks` call site inside `maskApi.ts`.

---

## 6. Config, dependencies, docs

- `requirements.txt`: drop `transformers`; add a comment block with
  `pip install -e C:\Users\WYK\Documents\HKUST\MPhil\Research\sam3` and the `hf auth login` step.
- `backend/config/server.json`: new `propagate` keys (`window_frames`, `overlap`, `anchor_max`,
  `job_ttl_seconds`), `sam3.checkpoint` / `sam3.bpe_path` / `sam3.max_resident_models`.
- `docs/USER_GUIDE.md` + `WORKFLOW_WALKTHROUGH.md`: the SAM 2/SAM 3 mode split is gone (phase 4).
- `README.md`: architecture section still claims the backend only decodes RLE — update in phase 4.

---

## 7. Tests

| Test                                   | Covers                                                                                                                                                                               |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `test/test_windows.py` (new)           | `plan_windows`: full coverage, overlap ≥ 2, stride, clamping at both clip ends, direction both runs forward first, single-frame ranges, anchor on the first/last frame               |
| `test/test_propagate_jobs.py` (new)    | Route behaviour with a **stub propagator** (no GPU): submit → poll → cancel, `since=` incremental masks, write policy counting, session-not-found, queue ordering, one job at a time |
| `test/test_sam3_segment.py` (new)      | Prompt routing (point/box/text) with a stub model, validation errors, instance splitting, coordinate handling                                                                        |
| `test/test_sessions_api.py` (existing) | Must stay green — sessions are untouched                                                                                                                                             |
| `test/test_sessions.py` (existing)     | Must stay green                                                                                                                                                                      |
| `frontend`                             | `node .\node_modules\typescript\bin\tsc -b` clean                                                                                                                                    |

The stub-predictor pattern is deliberately copied from the ref (`run_track` is documented as testable
with a stub), so the whole request sequence is covered without a GPU.

---

## 8. Phases and checkpoints

| Phase | Work                                                                                                        | Checkpoint                                                                                        |
| ----- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| **1** | `domain/windows.py`, `domain/mask_prompt.py`, `core/jobs.py`, schemas, routes, config; all tests with stubs | `pytest backend/test` green, no GPU needed                                                        |
| **2** | `inference/models.py`, `sam3_image.py`, `sam3_video.py`; wire deps + lifespan; delete the HF modules        | `/api/sam3/status` reports loaded models; a 3-frame propagate returns masks                       |
| **3** | Frontend: API clients, `Workspace` state rework, `Toolbar`, `PropagationPanel`, `TimelineStrip`             | Click → draft → accept; propagate to end with progress and cancel; correct → stale → re-propagate |
| **4** | Delete leftovers, update docs/README/config, final test pass                                                | `tsc -b` + pytest + a manual end-to-end clip                                                      |

Phases 1 and 2 are independent of the frontend, so the UI keeps working (on the old endpoints) until
phase 3 swaps the clients in one commit.

---

## 9. Risks and unknowns

| #   | Risk                                                                                                                                                   | Mitigation                                                                                                                                                                                        |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| R1  | **`sam3` is not installed in any environment on this machine** and the checkpoint is HF-gated — so nothing here can be smoke-tested until that is done | Do the install + `hf auth login` first; phase 2 is blocked on it. This is the single biggest risk to the schedule.                                                                                |
| R2  | `add_mask` with a **fresh** `obj_id` (no prior text/box prompt) registering a new object                                                               | Verified by reading `add_tracker_new_mask` (it creates a tracker state when the id is unknown); confirm with a 3-frame smoke test in phase 2, and fall back to a text/box prompt if it misbehaves |
| R3  | Two models resident → CUDA OOM                                                                                                                         | `sam3_max_resident_models=1` default, per-model device pins, explicit eviction                                                                                                                    |
| R4  | Point prompts: exact coordinate convention for `predict_inst` (`normalize_coords`)                                                                     | Determine from the example notebook during phase 2; the API takes pixels and the service converts                                                                                                 |
| R5  | Window chaining can drift across a boundary                                                                                                            | Overlap re-anchoring + `propagate_anchor_max` anchors; expose the window boundaries in the UI so a bad hand-off is visible (spec §7.4)                                                            |
| R6  | Long jobs (thousands of frames) block a single-GPU queue for a long time                                                                               | Cancellation between frames and windows; queue is visible; `To end` warns with the frame count                                                                                                    |
| R7  | Job results are large (RLE per frame)                                                                                                                  | `GET /jobs/{id}?since=` incremental reads; `include_masks=false` for status-only polls                                                                                                            |

---

## 10. Suggested splits (if you want less in one go)

- **Slice A (backend-only, no GPU):** phase 1. Everything testable today, and it de-risks the API
  shape before any model work.
- **Slice B:** phase 2, once `sam3` is installed.
- **Slice C:** phase 3 + 4 (frontend + cleanup).

Reply with the confirmations on §1 (C1–C6) and which slice to start with.
