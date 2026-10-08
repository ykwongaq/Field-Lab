import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Clip } from "../lib/clip";
import type { FrameSource } from "../lib/frames";
import { SessionFrameSource } from "../lib/frames";
import type { ZipArchive } from "../lib/zip";
import type { PromptPoint, RawRle } from "../types";
import { LabelStore } from "../lib/labelStore";
import { MODE_VOCABULARY } from "../lib/project";
import { NEW_TRACKLET_LABEL } from "../lib/clip";
import { rleArea } from "../lib/rle";
import {
    composeRle,
    polygonToRle,
    rleToDecoded,
    type FramePoint,
    type PaintMode,
} from "../lib/raster";
import { outlineFromRle, ringsToRle } from "../lib/contour";
import {
    fetchSam3Status,
    segmentFrame,
    type PromptBox,
    type PromptKind,
    type Sam3Status,
    type SegmentResult,
} from "../lib/sam3Api";
import {
    cancelPropagation,
    fetchPropagateStatus,
    getPropagationJob,
    startPropagation,
    type PropagateStatus,
    type PropagatedFrame,
} from "../lib/propagateApi";
import { usePanelLayout } from "../hooks/usePanelLayout";
import { useExport } from "../hooks/useExport";
import { useSaveProject } from "../hooks/useSaveProject";
import { useWorkspaceShortcuts } from "../hooks/useWorkspaceShortcuts";
import { useLabelEditing } from "../hooks/useLabelEditing";
import { VideoPanel } from "./VideoPanel";
import { isLive, type PropQueueEntry } from "./PropagationQueue";
import { TimelineStrip, type TimelineFrameState } from "./TimelineStrip";
import { TrackletList } from "./TrackletList";
import { LabelsPanel } from "./LabelsPanel";
import { LabelEditor } from "./LabelEditor";
import { Toolbar, type DrawMethod, type Tool } from "./Toolbar";
import { Button, Chip, Dialog, Icon, Splitter } from "../ui";
import { ExportMenu } from "./ExportMenu";
import styles from "./Workspace.module.css";

export interface WorkspaceNotice {
    kind: "success" | "info";
    text: string;
}

interface PropagateRun {
    /** Identifies the queued run, so a review can be matched back to its entry. */
    jobId: string;
    trackletId: number;
    anchor: number;
    first: number;
    last: number;
    backend: string;
    model: string;
    device: string;
    elapsedMs: number;
    /** Distinct frames produced so far, for the progress read-out. */
    framesDone: number;
    framesTotal: number;
    /** How the result lands: a refinement replaces, a first pass fills gaps. */
    writePolicy: "skip-existing" | "replace-range";
    /** Hand-verified frames that seeded this run, the anchor excluded. */
    pinned: number;
    /** True while the backend is still working through this run. */
    live: boolean;
    /** Window boundaries of the run, as frame indices (first window excluded). */
    windows: number[];
    masks: Map<number, PropagatedFrame>;
}

/**
 * The reviewer's view of one queued run.
 *
 * `masks` is handed over by reference on purpose: the queue mutates that map as
 * frames stream in and hands over a fresh run object each poll, which is what
 * makes the canvas repaint without re-downloading anything.
 */
function entryToRun(
    entry: PropQueueEntry,
    status: PropagateStatus | null,
): PropagateRun {
    return {
        jobId: entry.jobId,
        trackletId: entry.trackletId,
        anchor: entry.anchor,
        first: entry.first,
        last: entry.last,
        backend: "sam3",
        model: status?.model ?? "SAM 3",
        device: status?.device ?? "?",
        elapsedMs: entry.elapsedMs,
        framesDone: entry.framesDone,
        framesTotal: entry.framesTotal,
        writePolicy: entry.writePolicy,
        pinned: entry.pinned,
        live: isLive(entry),
        windows: entry.windows,
        masks: entry.masks,
    };
}

/**
 * What a finished run needs to be written into the clip.
 *
 * Both the live run under review and a queued batch entry satisfy it, so one
 * writer serves the single-object and the batch path.
 */
interface WritableRun {
    trackletId: number;
    /** The frame carrying the mask the run started from; never overwritten. */
    anchor: number;
    first: number;
    last: number;
    masks: Map<number, PropagatedFrame>;
}

/**
 * Write finished runs into the clip.
 *
 * A run replaces the range it covers: every frame it produced becomes the
 * object's mask there, and a frame it produced nothing for loses a derived mask
 * it used to hold — the tracker saw no object, so the old mask is gone. Two
 * frames are never touched: one a human verified (that is the input, not the
 * output) and the anchor (the mask the run started from).
 *
 * `partial` is a run that was stopped before it covered its range. It only adds
 * what it produced and removes nothing: the frames it never reached were never
 * looked at, so they still say what they said before.
 */
function writeRuns(
    clip: Clip,
    runs: WritableRun[],
    isVerified: (objectId: number, frame: number) => boolean,
    partial: boolean,
): Clip {
    let next = clip;
    for (const run of runs) {
        if (!next.tracklets.some((item) => item.id === run.trackletId))
            continue;
        const stored = new Set<number>();
        for (const mask of run.masks.values()) {
            if (mask.area <= 0) continue;
            if (isVerified(run.trackletId, mask.frameIndex)) continue;
            next = next.replaceMask(run.trackletId, mask.frameIndex, mask.rle);
            stored.add(mask.frameIndex);
        }
        if (partial) continue;
        // Re-read the tracklet: the writes above replaced it with a new object.
        const tracklet = next.tracklets.find(
            (item) => item.id === run.trackletId,
        );
        if (!tracklet) continue;
        for (let frame = run.first; frame <= run.last; frame++) {
            if (frame === run.anchor) continue;
            if (stored.has(frame)) continue;
            if (isVerified(run.trackletId, frame)) continue;
            if (next.rawMaskAt(tracklet, frame) === null) continue;
            next = next.removeMask(run.trackletId, frame);
        }
    }
    return next;
}

/**
 * The key every per-frame piece of state is held under.
 *
 * Drafts, staleness and provenance are all about one object on one frame, so they
 * share a single key shape and a single lookup.
 */
function frameKey(objectId: number, frame: number): string {
    return `${objectId}:${frame}`;
}

/**
 * The draft owner for a mask being *created* rather than edited.
 *
 * A new mask belongs to no object until it is accepted, so its drafts are keyed
 * under this sentinel. Tracklet ids are positive, so it can never collide.
 */
const NEW_OBJECT_ID = -1;

/** How often a running propagation is polled for newly produced masks. */
const PROPAGATE_POLL_MS = 300;

/** How long the "why is this greyed out" hint stays on screen, in ms. */
const PROP_HINT_MS = 4000;

/**
 * The most frames a playback queue may hold before the oldest are dropped.
 *
 * A run that produces faster than the clip plays must not build an unbounded
 * backlog: skipping frames keeps the playback near the front of the run.
 */
const PROP_PLAYBACK_MAX = 300;

/**
 * The draw methods, split by phase (spec P3: prompt creates, brush and polygon
 * correct — never mixed).
 *
 * `Add mask` always offers every method. `Edit mask` offers only the two that
 * correct an existing mask, *except* on a frame where the selected object has no
 * mask at all: there is nothing to correct there, so the create methods come back
 * and a redraw fills that frame of the same object instead of becoming a second
 * one.
 */
const CREATE_METHODS: DrawMethod[] = [
    "point",
    "box",
    "text",
    "polygon",
    "brush",
];
const CORRECT_METHODS: DrawMethod[] = ["polygon", "brush"];

/** Labels for the method strip. */
const METHOD_LABELS: Record<DrawMethod, string> = {
    point: "Point",
    box: "Box",
    text: "Text",
    polygon: "Polygon",
    brush: "Brush",
};

/** Wait, but give up promptly when the run is cancelled. */
function sleep(ms: number, signal?: AbortSignal): Promise<void> {
    return new Promise((resolve, reject) => {
        const timer = setTimeout(resolve, ms);
        signal?.addEventListener(
            "abort",
            () => {
                clearTimeout(timer);
                reject(new DOMException("Aborted", "AbortError"));
            },
            { once: true },
        );
    });
}

interface WorkspaceProps {
    clip: Clip;
    zip: ZipArchive;
    /** Where the displayed pixels come from; the archive only holds the rest. */
    frames: FrameSource;
    notice?: WorkspaceNotice | null;
    onDismissNotice?: () => void;
    onReset: () => void;
}

export function Workspace({
    clip: initialClip,
    zip,
    frames,
    notice: externalNotice = null,
    onDismissNotice,
    onReset,
}: WorkspaceProps) {
    // Labels carry the taxonomy, and saved overrides are keyed by label id.
    // Folding them in here means the list, inspector and export all read one
    // consistent clip from the first render.
    const storeRef = useRef<LabelStore | null>(null);
    const [clip, setClip] = useState(() => {
        const loaded = LabelStore.load(initialClip.name, initialClip);
        storeRef.current = loaded;
        return loaded.applyTo(initialClip);
    });
    const store = storeRef.current as LabelStore;
    const vocab = MODE_VOCABULARY[clip.mode];
    const [frameIndex, setFrameIndex] = useState(0);
    const [playing, setPlaying] = useState(false);
    const [selectedId, setSelectedId] = useState<number | null>(
        clip.tracklets[0]?.id ?? null,
    );
    const [maskOpacity, setMaskOpacity] = useState(0.55);
    // Bumped after a taxonomy edit so the (mutable) store's new value is read
    // again on the next render; the counter's own value is not needed.
    const [, setTick] = useState(0);

    // The two draggable seams and their persisted sizes live in a hook; the
    // workspace only attaches the refs and forwards the pointer deltas.
    const {
        bodyRef,
        sidebarRef,
        bodyStyle,
        sidebarStyle,
        onPanelDrag,
        onLabelsDrag,
        onPanelNudge,
        onLabelsNudge,
        resetPanelW,
        resetLabelsH,
    } = usePanelLayout();

    const semantic = clip.mode === "semantic";
    // There is one model now: SAM 3 answers text, box and point prompts, and the
    // same tracker propagates the mask it produced.
    const modelName = "SAM 3";
    const [tool, setTool] = useState<Tool>("review");
    // The method the user last picked. What is actually in force may be clamped
    // to the bar's available set — see `method`, derived below.
    const [preferredMethod, setMethod] = useState<DrawMethod>("point");
    const [paintMode, setPaintMode] = useState<PaintMode>("add");
    const [brushSize, setBrushSize] = useState(24);
    /**
     * Uncommitted drafts, keyed by `objectId:frame`.
     *
     * A draft is per frame, not global: navigating away from a frame with unsaved
     * work used to discard it silently, which is the one failure the interaction
     * spec singles out (G3, §6.2). A value may be `null`, which is a real state —
     * "everything was erased on this frame" — so `has()` is what distinguishes it
     * from "nothing was drawn here".
     */
    const [drafts, setDrafts] = useState<Map<string, RawRle | null>>(
        () => new Map(),
    );
    /** Undo stacks for those drafts, keyed the same way (depth 40 each). */
    const [histories, setHistories] = useState<Map<string, (RawRle | null)[]>>(
        () => new Map(),
    );
    const [polygon, setPolygon] = useState<FramePoint[]>([]);
    /**
     * The draft a live outline edit belongs to, with the rings the user has
     * dragged. Claiming the new RLE in the same update as it is pushed is what
     * keeps the handles still: they are the source of truth, not the raster.
     */
    const [draftRings, setDraftRings] = useState<{
        of: RawRle;
        rings: FramePoint[][];
    } | null>(null);

    const [prompt, setPrompt] = useState<PromptPoint[]>([]);
    const [box, setBox] = useState<PromptBox | null>(null);
    const [candidate, setCandidate] = useState<SegmentResult | null>(null);
    const [segmenting, setSegmenting] = useState(false);
    const [promptError, setPromptError] = useState<string | null>(null);
    const segmentAbortRef = useRef<AbortController | null>(null);
    const [sam, setSam] = useState<Sam3Status | null>(null);

    const [className, setClassName] = useState("");
    /**
     * The object an open delete confirmation will remove, pending confirmation.
     *
     * Removing an object is not undoable, so both the list's trash button and the
     * mask bar's “delete the only mask” route open the one modal shell rather than
     * deleting on the click. `then` is the tool to fall back to when the object
     * removed is the one Edit mask is attached to.
     */
    const [pendingDelete, setPendingDelete] = useState<{
        id: number;
        then: Tool;
    } | null>(null);
    // The workspace shows no action toasts of its own, except the outcome of a
    // save or a propagation run: the shell's notice (currently the frame-list
    // mismatch warning) shows through unless one of those has something of its
    // own to report.
    const [saveNotice, setSaveNotice] = useState<WorkspaceNotice | null>(null);
    /**
     * The `clip.editCount` the last save captured.
     *
     * The leave-guard compares against it, so a saved project stops prompting on
     * unload; the next edit pushes `editCount` past it and the guard arms again.
     */
    const [savedEditCount, setSavedEditCount] = useState(0);
    const notice = saveNotice ?? externalNotice;

    // The export chooser's state, runner and per-project options. It is created
    // here, ahead of the keyboard effect that reads `exportOpen`.
    const {
        exportOpen,
        exportBusyId,
        exportProgress,
        exportError,
        exportOptions,
        openExport,
        closeExport,
    } = useExport({
        clip,
        frames,
        zip,
        store,
    });

    // "Save" packs the current state back into a `.project` archive and
    // downloads it, so the annotated result can be reopened later.
    const { saveBusy, saveProgress, saveProject } = useSaveProject({
        clip,
        zip,
        store,
    });

    const [propStatus, setPropStatus] = useState<PropagateStatus | null>(null);
    // The frame range a run covers, defaulting to the whole clip. The two bars
    // on the timeline narrow it, and the anchor always stays inside.
    const [propFirst, setPropFirst] = useState(0);
    const [propLast, setPropLast] = useState(() =>
        Math.max(0, clip.frameCount - 1),
    );
    const [propagating, setPropagating] = useState(false);
    const [propError, setPropError] = useState<string | null>(null);
    const [propRun, setPropRun] = useState<PropagateRun | null>(null);
    const propAbortRef = useRef<AbortController | null>(null);
    //: Every job of the current batch, so Stop can stop all of them and not just
    //: the one being watched.
    const propJobRef = useRef<string[]>([]);
    /**
     * Frames waiting to be drawn, and the job they belong to.
     *
     * A window's masks can land in one burst, and the reviewer wants to watch the
     * run fill the clip in rather than be shown its last frame. The poll loop
     * appends what it receives here and a frame clock draws them one at a time, so
     * propagation plays back however the backend batches it.
     */
    const propPlaybackRef = useRef<number[]>([]);
    const propPlaybackJobRef = useRef<string | null>(null);
    /**
     * The transient "why is this greyed out" hint, and the button it points at.
     *
     * The coordinates are viewport-relative, because the hint is laid out with
     * `position: fixed` — the tool bar clips its own overflow, so a box nested
     * inside it would be cut off. The frame and tool it was raised for travel
     * with it, so stepping away retires it without an effect.
     */
    const [propHint, setPropHint] = useState<{
        x: number;
        y: number;
        frame: number;
        tool: Tool;
    } | null>(null);
    const propButtonRef = useRef<HTMLButtonElement | null>(null);
    // It is transient: it answers the click, then gets out of the way.
    useEffect(() => {
        if (!propHint) return;
        const timer = window.setTimeout(() => setPropHint(null), PROP_HINT_MS);
        return () => window.clearTimeout(timer);
    }, [propHint]);
    /**
     * Objects queued for propagation.
     *
     * Empty means "just the selected object", which keeps one-click propagation
     * working. With a batch, each object is a separate job run one after another.
     */
    const [propBatch, setPropBatch] = useState<number[]>([]);
    const [propQueue, setPropQueue] = useState<PropQueueEntry[]>([]);
    /**
     * `${objectId}:${frame}` for every frame a human drew, corrected or cleared.
     *
     * This is the frame's *provenance*: someone looked at it. Only such a frame may
     * anchor a run, and no run may overwrite one. A frame that came out of a
     * propagation and was accepted in bulk is deliberately absent — nobody checked
     * it — and that distinction is what the refinement loop turns on.
     */
    const [verifiedFrames, setVerifiedFrames] = useState<Set<string>>(
        () => new Set(),
    );
    const dismissNotice = useCallback(() => {
        setSaveNotice(null);
        onDismissNotice?.();
    }, [onDismissNotice]);

    const fetchStatus = fetchSam3Status;
    const refreshSam = useCallback(() => {
        setSam(null);
        setPropStatus(null);
        void fetchStatus().then(setSam);
        void fetchPropagateStatus().then(setPropStatus);
    }, [fetchStatus]);
    useEffect(() => {
        const controller = new AbortController();
        void fetchStatus(controller.signal).then(setSam, () => {});
        void fetchPropagateStatus(controller.signal).then(
            setPropStatus,
            () => {},
        );
        return () => controller.abort();
    }, [fetchStatus]);

    useEffect(() => {
        // An uncommitted draft is as worth guarding as a committed edit — but
        // work already written to a file is not unsaved, so the guard only arms
        // for edits made since the last save.
        if (clip.editCount === savedEditCount && drafts.size === 0) return;
        const onBeforeUnload = (event: BeforeUnloadEvent) => {
            event.preventDefault();
        };
        window.addEventListener("beforeunload", onBeforeUnload);
        return () => window.removeEventListener("beforeunload", onBeforeUnload);
    }, [clip.editCount, drafts.size, savedEditCount]);

    const refresh = useCallback(() => setTick((value) => value + 1), []);

    const selected = useMemo(
        () =>
            clip.tracklets.find((tracklet) => tracklet.id === selectedId) ??
            null,
        [clip.tracklets, selectedId],
    );

    const stepFrame = useCallback(
        (delta: number) =>
            setFrameIndex((index) =>
                Math.min(clip.frameCount - 1, Math.max(0, index + delta)),
            ),
        [clip.frameCount],
    );

    const togglePlay = useCallback(() => setPlaying((value) => !value), []);

    const selectTracklet = useCallback(
        (id: number) => {
            setSelectedId(id);
            const tracklet = clip.tracklets.find(
                (candidate) => candidate.id === id,
            );
            if (tracklet && tracklet.maskFrames.first >= 0) {
                setFrameIndex(tracklet.maskFrames.first);
            }
        },
        [clip.tracklets],
    );

    const selectOnCanvas = useCallback((id: number) => {
        setSelectedId(id);
    }, []);

    const selectedLabel = selected ? clip.labelFor(selected) : null;

    // Labels: the editor, the picker's "＋" target, the pending delete and the
    // per-object assignment all live in a hook.
    const {
        newLabelId,
        setNewLabelId,
        labelEditor,
        setLabelEditor,
        pendingLabelDelete,
        setPendingLabelDelete,
        editorLabel,
        editorTaxonomy,
        editorColor,
        pendingLabel,
        pendingLabelCount,
        assignTrackletLabel,
        createLabelForTracklet,
        openLabelEditor,
        saveLabel,
        deleteLabel,
        selectLabel,
    } = useLabelEditing({
        clip,
        setClip,
        store,
        refresh,
        selectTracklet,
    });

    const resetPrompt = useCallback(() => {
        segmentAbortRef.current?.abort();
        segmentAbortRef.current = null;
        setPrompt([]);
        // The box belongs to the prompt, exactly like the clicks do. Leaving it
        // behind meant the next point prompt was sent as "this point AND that
        // box", so the box won and the model kept returning the first object.
        setBox(null);
        setCandidate(null);
        setSegmenting(false);
        setPromptError(null);
    }, []);

    /** The committed mask an edit started from, so a redraw reads as a change. */
    const originalMask = useMemo(
        () =>
            tool === "editMask" && selected
                ? clip.rawMaskAt(selected, frameIndex)
                : null,
        [tool, selected, clip, frameIndex],
    );

    /**
     * Which object the current frame's draft belongs to.
     *
     * Editing attaches to the selected object; creating a mask attaches to nothing
     * yet, so it uses the sentinel until Accept gives it an id.
     */
    const draftOwnerId =
        tool === "editMask" ? (selectedId ?? NEW_OBJECT_ID) : NEW_OBJECT_ID;
    const draftKey = frameKey(draftOwnerId, frameIndex);
    /**
     * The draft to show for this frame.
     *
     * A stored draft wins. Failing that, Edit mask seeds itself from the mask
     * already on the frame — seeded, not stored, so merely visiting a frame does
     * not mark it as carrying unsaved work.
     */
    const draft = drafts.has(draftKey)
        ? (drafts.get(draftKey) ?? null)
        : tool === "editMask" && selected
          ? clip.rawMaskAt(selected, frameIndex)
          : null;
    const draftHistory = histories.get(draftKey) ?? [];

    // The draft callbacks stay stable across renders, so they read the live values
    // through refs rather than closing over one render's snapshot.
    const draftRef = useRef(draft);
    draftRef.current = draft;
    const draftKeyRef = useRef(draftKey);
    draftKeyRef.current = draftKey;

    /** Drop this frame's draft; its undo stack goes with it. */
    const discardDraft = useCallback(() => {
        const key = draftKeyRef.current;
        setDrafts((current) => {
            if (!current.has(key)) return current;
            const next = new Map(current);
            next.delete(key);
            return next;
        });
        setHistories((current) => {
            if (!current.has(key)) return current;
            const next = new Map(current);
            next.delete(key);
            return next;
        });
        resetPrompt();
        setPolygon([]);
        setDraftRings(null);
    }, [resetPrompt]);

    /** Replace this frame's draft, pushing the old one onto its undo stack. */
    const pushDraft = useCallback((next: RawRle | null) => {
        const key = draftKeyRef.current;
        setHistories((current) => {
            const stack = [...(current.get(key) ?? []), draftRef.current].slice(
                -40,
            );
            const map = new Map(current);
            map.set(key, stack);
            return map;
        });
        setDrafts((current) => new Map(current).set(key, next));
    }, []);

    /**
     * Clear the transient prompt state (clicks, polygon) without touching drafts.
     *
     * A prompt belongs to the frame and the tool it was made with; the drawn work
     * does not. That split is what lets a tool switch or a frame change preserve the
     * draft (§6.1) while still starting the next prompt clean.
     */
    const resetTransient = useCallback(() => {
        resetPrompt();
        setPolygon([]);
        setDraftRings(null);
    }, [resetPrompt]);

    const targetClass = useMemo(
        () => (semantic ? clip.findTrackletByLabel(className) : null),
        [semantic, clip, className],
    );
    const classLabels = useMemo(
        () => [...new Set(clip.tracklets.map((t) => t.label))].sort(),
        [clip.tracklets],
    );

    /** True when a human drew, corrected or cleared this frame. */
    const isVerified = useCallback(
        (objectId: number, frame: number) =>
            verifiedFrames.has(frameKey(objectId, frame)),
        [verifiedFrames],
    );

    /**
     * Record that a human has just looked at frame `k` of an object.
     *
     * That is the whole of it: the frame joins the *human* set, so it is painted in
     * the label colour on the timeline, no later run may overwrite it, and it
     * seeds the next run as a conditioning mask.
     */
    const markCorrected = useCallback((objectId: number, frame: number) => {
        setVerifiedFrames((current) =>
            new Set(current).add(frameKey(objectId, frame)),
        );
    }, []);

    /**
     * The human frames that seed a run.
     *
     * Every mask a person has drawn or corrected inside the range, the anchor
     * excepted (it travels as the run's own anchor mask). Model output is never
     * sent: a window conditioned on its own guess keeps its mistakes, which is
     * the whole reason a re-run is seeded from human work.
     */
    const pinsFor = useCallback(
        (objectId: number, first: number, last: number, anchor: number) => {
            const tracklet = clip.tracklets.find(
                (item) => item.id === objectId,
            );
            const pins = new Map<number, RawRle>();
            if (!tracklet || tracklet.maskFrames.first < 0) return pins;
            const start = Math.max(first, tracklet.maskFrames.first);
            const end = Math.min(last, tracklet.maskFrames.last);
            for (let frame = start; frame <= end; frame++) {
                if (frame === anchor) continue;
                if (!isVerified(objectId, frame)) continue;
                const mask = clip.rawMaskAt(tracklet, frame);
                if (mask) pins.set(frame, mask);
            }
            return pins;
        },
        [clip, isVerified],
    );

    const samAvailable = sam?.available ?? false;

    const selectedHasMaskHere =
        selected !== null && clip.rawMaskAt(selected, frameIndex) !== null;

    /**
     * The mask a *human* drew or corrected on a frame, if any.
     *
     * Only such a mask may seed a run: the anchor is written into tracker memory as
     * ground truth, so anchoring on a frame the tracker itself produced is the one
     * way to make a run carry its own mistake forward. A *cleared* frame is excluded
     * — "the object is not here" is a verification, but it is not conditionable.
     */
    const humanMaskAt = useCallback(
        (tracklet: Clip["tracklets"][number], frame: number): RawRle | null =>
            verifiedFrames.has(frameKey(tracklet.id, frame))
                ? clip.rawMaskAt(tracklet, frame)
                : null,
        [verifiedFrames, clip],
    );
    /**
     * Whether this frame may start a run.
     *
     * The anchor is the frame a run is conditioned on, and only human input is
     * trustworthy enough for that, so a run starts where the reviewer has drawn or
     * corrected — and the range always holds at least that one human frame.
     */
    const selectedHasHumanMaskHere =
        selected !== null && humanMaskAt(selected, frameIndex) !== null;

    /**
     * Whether the create methods belong in the bar right now.
     *
     * `Add mask` always creates. In instance mode `Edit mask` is correction only
     * — unless the selected object has no mask on this frame, where the create
     * methods are the only meaningful gesture: they fill *this* object's frame
     * (Accept replaces its mask) rather than making a second object for the same
     * thing. Semantic mode keeps its create path in `Add mask`, where the
     * class-name box lives.
     */
    const canCreateHere =
        tool === "addMask" || (!semantic && !selectedHasMaskHere);
    const availableMethods = useMemo<DrawMethod[]>(
        () => (canCreateHere ? CREATE_METHODS : CORRECT_METHODS),
        [canCreateHere],
    );

    /**
     * The method actually in force.
     *
     * The bar's set can shrink when the frame or the selection changes, so a
     * preference that is no longer offered falls back to the first one that is.
     * Deriving it here rather than writing it back to state keeps the strip, the
     * pointer handler, the commit path and the status line on one value by
     * construction — there is no render in which they can disagree.
     */
    const method: DrawMethod = availableMethods.includes(preferredMethod)
        ? preferredMethod
        : availableMethods[0];

    /**
     * The editable outline of the mask on this frame, or null when Polygon is
     * drawing a new shape rather than correcting one.
     *
     * It is derived from the *draft*, so strokes made before switching to Polygon
     * are part of the shape. Requiring a committed mask on the frame (rather than
     * just a draft) is what keeps the two polygon behaviours apart: with nothing
     * to correct the tool stays click-to-place, and drawing a shape must not flip
     * it into outline editing halfway through.
     */
    const outlineRings = useMemo(() => {
        if (tool !== "editMask" || method !== "polygon") return null;
        if (!selectedHasMaskHere || !draft) return null;
        if (draftRings && draftRings.of === draft) return draftRings.rings;
        return outlineFromRle(draft);
    }, [tool, method, selectedHasMaskHere, draft, draftRings]);

    /**
     * Bake edited rings back into the draft.
     *
     * The new RLE is claimed by `draftRings` in the same update as it is pushed,
     * so the handles stay exactly where the user left them instead of being
     * re-derived — and re-simplified — from the raster on the next render.
     */
    const commitOutline = useCallback(
        (rings: FramePoint[][]) => {
            const rle = ringsToRle(rings, clip.width, clip.height);
            // Rings that enclose no pixel are not a mask. Leave the draft alone
            // rather than erasing: `Delete mask on this frame` is that gesture.
            if (!rle) return;
            setDraftRings({ of: rle, rings });
            pushDraft(rle);
        },
        [clip.width, clip.height, pushDraft],
    );

    const propRunRef = useRef(propRun);
    propRunRef.current = propRun;
    //: Assigned once `acceptPropagation` exists (below). `changeTool` commits a
    //: finished run through it, so leaving the tool keeps the run, not drops it.
    const acceptPropagationRef = useRef<() => void>(() => {});

    const discardPropagation = useCallback((restoreFrame = false) => {
        propAbortRef.current?.abort();
        propAbortRef.current = null;
        propJobRef.current = [];
        propPlaybackRef.current = [];
        propPlaybackJobRef.current = null;
        if (restoreFrame && propRunRef.current)
            setFrameIndex(propRunRef.current.anchor);
        setPropagating(false);
        setPropRun(null);
        setPropQueue([]);
        setPropError(null);
    }, []);

    const changeTool = useCallback(
        (next: Tool) => {
            if (next === "editMask" && !selected) return;
            if (next === "propagate" && !selectedHasHumanMaskHere) return;
            if (next !== "review") {
                setPlaying(false);
            }
            if (next === "propagate" && !propStatus) {
                void fetchPropagateStatus().then(setPropStatus);
            }
            // A finished run is committed on the way out, never thrown away: the
            // masks a reviewer just watched are work, and leaving the tool is not a
            // reason to lose them.
            if (propRunRef.current) acceptPropagationRef.current();
            else discardPropagation(true);
            setTool(next);
            setPaintMode("add");
            // Drafts survive a tool switch (§6.1): the prompt is what is specific to
            // a tool, not the work already drawn.
            resetTransient();
        },
        [
            selected,
            selectedHasHumanMaskHere,
            samAvailable,
            propStatus,
            discardPropagation,
            resetTransient,
        ],
    );

    /**
     * Delete the selected object's mask on this frame, as a deliberate act.
     *
     * This is the "that mask is wrong, draw it again" gesture, and it is not
     * merely tidier than painting over the mask: with Edit mask the model's result
     * is *composed* into the existing draft with `add`, so a click on top of a bad
     * mask can only ever grow it. Clearing first is what makes the replacement a
     * fresh prediction instead of a union with the mask being rejected.
     *
     * Only reachable while the object keeps another mask. When this is its *only*
     * mask, clearing it would delete the object itself (`removeMask` drops a
     * tracklet left with no masks), so the bar routes that click through the
     * delete confirmation instead.
     */
    const clearMaskOnFrame = useCallback(() => {
        if (selectedId === null || !selected) return;
        if (clip.rawMaskAt(selected, frameIndex) === null) return;
        // Defensive: the bar sends the single-mask case to the confirmation.
        if (selected.maskFrames.count <= 1) {
            return;
        }
        setClip(clip.removeMask(selectedId, frameIndex));
        markCorrected(selectedId, frameIndex);
        refresh();
        // Stay in Edit mask with an explicitly empty draft. The redraw then
        // *replaces* the mask on this frame; going through Add mask instead would
        // create a second object for the same thing, which is not a correction.
        // Stored on purpose: "erased on this frame" is itself uncommitted work, and
        // the redraw must compose onto nothing rather than onto the mask rejected.
        setDrafts((current) =>
            new Map(current).set(frameKey(selectedId, frameIndex), null),
        );
        resetTransient();
        setTool("editMask");
    }, [
        selectedId,
        selected,
        clip,
        frameIndex,
        markCorrected,
        refresh,
        resetTransient,
    ]);

    const toolRef = useRef(tool);
    toolRef.current = tool;
    // Moving between frames keeps every draft (§6.2); only the prompt resets, since
    // it is about the frame it was made on.
    useEffect(() => {
        if (toolRef.current !== "review" && toolRef.current !== "propagate")
            resetTransient();
    }, [frameIndex, resetTransient]);
    const discardPropagationRef = useRef(discardPropagation);
    discardPropagationRef.current = discardPropagation;
    // Switching object keeps drafts — they are keyed per object — and drops the run
    // that belonged to the old one. Edit mask also clears its prompt, which named
    // the object being edited; a create prompt is left alone.
    useEffect(() => {
        if (toolRef.current === "editMask") resetTransient();
        if (toolRef.current === "propagate") {
            // Switching object keeps the run that is in hand by committing it.
            if (propRunRef.current) acceptPropagationRef.current();
            else discardPropagationRef.current();
        }
    }, [selectedId, resetTransient]);

    const changeMethod = useCallback(
        (next: DrawMethod) => {
            if (next === method) return;
            // A method the bar is not offering is refused here too, so the
            // keyboard letters (P / B / …) cannot select a button that is not on
            // screen — an Edit-mask frame with a mask has no Point/Box/Text.
            if (!availableMethods.includes(next)) return;
            const needsModel =
                next === "point" || next === "box" || next === "text";
            resetPrompt();
            setPolygon([]);
            setMethod(next);
            // The Add/Erase toggle is only offered for Brush and Polygon. For the
            // model methods exclusion is expressed as Shift-click, so a paint mode
            // left over from an earlier brush would silently *subtract* the model's
            // result with no visible control to explain why.
            if (needsModel) setPaintMode("add");
            // Selecting a model method while SAM 3 is down used to return
            // silently, which reads as a broken button. Select it anyway and say
            // why it cannot run; Polygon and Brush keep working regardless.
            setPromptError(
                needsModel && !samAvailable
                    ? `${modelName} is unavailable${sam?.error ? `: ${sam.error}` : ""}. Polygon and Brush still work; check the backend status to fix this.`
                    : null,
            );
        },
        [method, availableMethods, samAvailable, resetPrompt, sam],
    );

    /**
     * Ask SAM 3 for a mask on the current frame.
     *
     * The kind is passed explicitly rather than inferred from the payload: a
     * click means "this object" and text means "this class", and those are
     * different questions that the backend answers with different code paths.
     */
    const segment = useCallback(
        async (request: {
            kind: PromptKind;
            points?: PromptPoint[];
            boxes?: PromptBox[];
            text?: string;
        }) => {
            const sessionId =
                frames instanceof SessionFrameSource ? frames.sessionId : null;
            if (!sessionId) {
                setPromptError(
                    "This clip is not open in a backend session, so SAM 3 cannot be used.",
                );
                return;
            }
            segmentAbortRef.current?.abort();
            const controller = new AbortController();
            segmentAbortRef.current = controller;
            setSegmenting(true);
            setPromptError(null);
            try {
                const result = await segmentFrame({
                    sessionId,
                    frameIndex,
                    // Every proposal is kept: a text prompt's instances become
                    // separate objects in an instance project, and a point/box
                    // prompt's candidates are the alternatives for one object.
                    maxInstances: 0,
                    signal: controller.signal,
                    ...request,
                });
                if (controller.signal.aborted) return;
                setCandidate(result);
            } catch (cause) {
                if (controller.signal.aborted) return;
                setCandidate(null);
                setPromptError(
                    cause instanceof Error ? cause.message : String(cause),
                );
            } finally {
                if (segmentAbortRef.current === controller) {
                    segmentAbortRef.current = null;
                    setSegmenting(false);
                }
            }
        },
        [frames, frameIndex, semantic],
    );

    const segmentPoints = useCallback(
        (points: PromptPoint[]) => {
            // A box already drawn on this frame belongs to the same prompt: it is
            // how the user says "this object here, but not that part".
            void segment({ kind: "point", points, boxes: box ? [box] : [] });
        },
        [segment, box],
    );

    const runTextPrompt = useCallback(() => {
        if (method !== "text" || !className.trim()) return;
        void segment({ kind: "text", text: className });
    }, [method, className, segment]);

    const segmentBox = useCallback(
        (rect: PromptBox) => {
            setBox(rect);
            void segment({ kind: "box", boxes: [rect] });
        },
        [segment],
    );

    const addPromptPoint = useCallback(
        (point: PromptPoint) => {
            const next = [...prompt, point];
            setPrompt(next);
            segmentPoints(next);
        },
        [prompt, segmentPoints],
    );

    const undoPromptPoint = useCallback(() => {
        const next = prompt.slice(0, -1);
        setPrompt(next);
        if (next.length === 0) {
            segmentAbortRef.current?.abort();
            segmentAbortRef.current = null;
            setCandidate(null);
            setSegmenting(false);
            setPromptError(null);
        } else {
            segmentPoints(next);
        }
    }, [prompt, segmentPoints]);

    const addPolygonPoint = useCallback((point: FramePoint) => {
        setPolygon((current) => [...current, point]);
    }, []);

    const closePolygon = useCallback(() => {
        const points = polygon.filter(
            (p, i) =>
                i === 0 || p.x !== polygon[i - 1].x || p.y !== polygon[i - 1].y,
        );
        if (points.length < 3) return;
        const rle = polygonToRle(points, clip.width, clip.height);
        setPolygon([]);
        if (!rle) {
            setPromptError("The polygon covers no pixel.");
            return;
        }
        setPromptError(null);
        pushDraft(composeRle(draft, rle, paintMode));
    }, [polygon, clip.width, clip.height, draft, paintMode, pushDraft]);

    const applyStroke = useCallback(
        (stroke: RawRle, mode: PaintMode) => {
            setPromptError(null);
            pushDraft(composeRle(draft, stroke, mode));
        },
        [draft, pushDraft],
    );

    const undo = useCallback(() => {
        if ((method === "point" || method === "box") && prompt.length > 0) {
            undoPromptPoint();
            return;
        }
        if (method === "polygon" && polygon.length > 0) {
            setPolygon((current) => current.slice(0, -1));
            return;
        }
        if (draftHistory.length === 0) return;
        const key = draftKeyRef.current;
        setDrafts((current) =>
            new Map(current).set(key, draftHistory[draftHistory.length - 1]),
        );
        setHistories((current) =>
            new Map(current).set(key, draftHistory.slice(0, -1)),
        );
    }, [method, prompt.length, undoPromptPoint, polygon.length, draftHistory]);

    const canUndo =
        ((method === "point" || method === "box") && prompt.length > 0) ||
        (method === "polygon" && polygon.length > 0) ||
        draftHistory.length > 0;

    /**
     * A text prompt in an instance project that matched several objects.
     *
     * Those are different objects, so the union of their masks must never become
     * one tracklet — that would be a single object holding several disjoint
     * blobs. They become separate tracklets instead (`commitInstances`).
     */
    const isTextSplit = Boolean(
        !semantic &&
        method === "text" &&
        candidate &&
        candidate.instances.length > 1,
    );

    const finalMask = useMemo(() => {
        if (isTextSplit) return null;
        // Every model method folds its candidate into the draft, so the commit
        // button is the *single* accept step. Leaving `text` out meant a text
        // prompt that matched one object produced a disabled commit button and
        // forced an extra "Apply" click first.
        const withCandidate =
            (method === "point" || method === "box" || method === "text") &&
            candidate &&
            candidate.area > 0
                ? composeRle(draft, candidate.rle, paintMode)
                : draft;
        return withCandidate && rleArea(withCandidate) > 0
            ? withCandidate
            : null;
    }, [isTextSplit, method, candidate, draft, paintMode]);
    const draftDecoded = useMemo(
        () => (draft ? rleToDecoded(draft) : null),
        [draft],
    );
    const draftArea = useMemo(() => (draft ? rleArea(draft) : 0), [draft]);
    // Identity against the committed mask is what "changed" means: undoing back to
    // the mask an edit started from reads as no change, exactly as it did before
    // drafts became per-frame.
    const draftChanged = draft !== originalMask || (candidate?.area ?? 0) > 0;

    /**
     * Turn the instances of one prompt into objects.
     *
     * `all` creates one tracklet per detected object; `largest` takes only the
     * biggest, which is what a reviewer wants when a text prompt also caught
     * clutter. Each tracklet gets its own id and category, so labels can be set
     * per object afterwards.
     */
    const commitInstances = useCallback(
        (selection: "all" | "largest") => {
            if (!candidate || tool === "review" || segmenting) return;
            const pool = candidate.instances;
            if (pool.length === 0) return;
            const chosen =
                selection === "all"
                    ? [...pool]
                    : [...pool].sort((a, b) => b.area - a.area).slice(0, 1);

            const label = className.trim() || NEW_TRACKLET_LABEL;
            const chosenLabel = clip.labelById(newLabelId);
            let next = clip;
            const created: number[] = [];
            for (const instance of chosen) {
                const added = next.addTracklet(frameIndex, instance.rle, label);
                next = chosenLabel
                    ? added.clip.setLabel(added.tracklet.id, chosenLabel.id)
                    : added.clip;
                created.push(added.tracklet.id);
                // The prompt was a person's: this first mask is human input, not
                // machine output, so the timeline paints it in the label colour and
                // a later run may anchor on it.
                markCorrected(added.tracklet.id, frameIndex);
            }
            setClip(next);
            setSelectedId(created[created.length - 1] ?? null);
            refresh();
            discardDraft();
            // Creating objects is an Add-mask act, so stay put and let the next
            // one be drawn straight away, whichever create tool the prompt was
            // fired from. Switching tools is the user's call, never the app's.
        },
        [
            candidate,
            tool,
            segmenting,
            clip,
            frameIndex,
            className,
            newLabelId,
            refresh,
            discardDraft,
            markCorrected,
        ],
    );

    const commitMask = useCallback(() => {
        if (tool === "review" || segmenting) return;

        // ----- Edit mask
        if (tool === "editMask") {
            if (selectedId === null || !selected || !draftChanged) return;
            const next = clip.replaceMask(selectedId, frameIndex, finalMask);
            const stillThere = next.tracklets.some((t) => t.id === selectedId);
            if (!stillThere) {
                const position = clip.tracklets.findIndex(
                    (t) => t.id === selectedId,
                );
                const fallback =
                    next.tracklets[
                        Math.min(position, next.tracklets.length - 1)
                    ] ?? null;
                setSelectedId(fallback?.id ?? null);
            }
            setClip(next);
            refresh();
            if (stillThere) markCorrected(selectedId, frameIndex);
            discardDraft();
            // Stay in Edit mask so the next frame of the same object can be
            // corrected straight away. Only when the edit removed the object
            // itself (an empty redraw on a one-mask object) is here no longer
            // valid: the selection falls back and Select is the safe landing.
            if (!stillThere) setTool("review");
            return;
        }

        // ----- Add mask
        if (!finalMask) return;
        if (!semantic) {
            // A text prompt names the concept, so let it name the object too
            // (commitInstances does the same for a multi-object prompt); an
            // explicit pick from the label dropdown still wins below.
            const newLabel = method === "text" ? className : NEW_TRACKLET_LABEL;
            const { clip: next, tracklet } = clip.addTracklet(
                frameIndex,
                finalMask,
                newLabel,
            );
            const chosenLabel = clip.labelById(newLabelId);
            setClip(
                chosenLabel ? next.setLabel(tracklet.id, chosenLabel.id) : next,
            );
            setSelectedId(tracklet.id);
            // The drawing is the human input this object is built on.
            markCorrected(tracklet.id, frameIndex);
            discardDraft();
            // Stay in Add mask. The next object is usually drawn immediately, and
            // returning to Select forced a click on the rail for every one.
            // `discardDraft` already cleared the draft, the clicks and the
            // polygon, so the bar is back to a clean prompt.
            return;
        }

        const label = className.trim() || NEW_TRACKLET_LABEL;
        try {
            if (targetClass) {
                const next = clip.paintClass(
                    frameIndex,
                    targetClass.id,
                    finalMask,
                );
                setClip(next);
                setSelectedId(targetClass.id);
                markCorrected(targetClass.id, frameIndex);
            } else {
                const { clip: next, tracklet } = clip.addClass(
                    frameIndex,
                    finalMask,
                    label,
                );
                setClip(next);
                setSelectedId(tracklet.id);
                markCorrected(tracklet.id, frameIndex);
            }
        } catch (cause) {
            setPromptError(
                cause instanceof Error ? cause.message : String(cause),
            );
            return;
        }
        discardDraft();
        // Same as the instance path: stay in Add mask for the next class or
        // object. The class name is left in the box so a repeat is one click.
    }, [
        tool,
        segmenting,
        selectedId,
        selected,
        draftChanged,
        clip,
        frameIndex,
        finalMask,
        store,
        refresh,
        discardDraft,
        markCorrected,
        semantic,
        method,
        candidate,
        className,
        targetClass,
        newLabelId,
    ]);

    /**
     * Remove an object and every mask it owns, by id.
     *
     * The object list's row trash asks for confirmation first, then routes
     * through here. The frame-level “delete mask” gesture is separate —
     * `clearMaskOnFrame` in the mask bar.
     */
    const deleteTracklet = useCallback(
        (id: number) => {
            const before = clip.tracklets.find((t) => t.id === id);
            if (!before) return;
            const next = clip.removeTracklet(id);
            if (next === clip) return;
            const position = clip.tracklets.findIndex((t) => t.id === id);
            const fallback =
                next.tracklets[Math.min(position, next.tracklets.length - 1)] ??
                null;
            // A row can be deleted while a different object is selected, so
            // only move the selection when the deleted object held it.
            setSelectedId((current) =>
                current === id ? (fallback?.id ?? null) : current,
            );
            // The object is gone, so every propagation trace of it must go too.
            // A run under review (or queued) belonged to this object, and leaving
            // it in state would paint its frames onto whatever object is created
            // or selected next — e.g. a new one-frame tracklet that then reads as
            // the deleted object's propagated result.
            if (propRunRef.current?.trackletId === id) discardPropagation();
            setPropQueue((current) =>
                current.filter((entry) => entry.trackletId !== id),
            );
            setPropBatch((current) => current.filter((item) => item !== id));
            setClip(next);
            refresh();
        },
        [clip, refresh, discardPropagation],
    );

    /** The object named by the open delete confirmation, if any. */
    const pendingDeleteTarget = useMemo(
        () =>
            pendingDelete === null
                ? null
                : (clip.tracklets.find((t) => t.id === pendingDelete.id) ??
                  null),
        [clip.tracklets, pendingDelete],
    );

    const propagateModel = "SAM 3 tracker";
    const propagateAvailable = propStatus?.available ?? false;
    const propagateNote = propStatus?.available
        ? null
        : `The tracker is unavailable (${propStatus?.error ?? "unknown reason"}).`;

    const propAnchor = propRun ? propRun.anchor : frameIndex;

    /**
     * The range a run will cover, as absolute frame indices.
     *
     * The anchor is clamped inside it: it holds the mask being propagated, so a
     * run over a range that excluded it could not start. The backend splits a long
     * range into overlapping windows, so the whole clip is a valid range.
     */
    const propRange = useMemo(() => {
        const first = Math.max(0, Math.min(propFirst, propAnchor));
        const last = Math.min(
            clip.frameCount - 1,
            Math.max(propLast, propAnchor),
        );
        return {
            back: propAnchor - first,
            forward: last - propAnchor,
            first,
            last,
        };
    }, [propFirst, propLast, propAnchor, clip.frameCount]);

    /**
     * Set the range from the timeline's two bars.
     *
     * The anchor stays inside — a range that excluded it could not seed a run —
     * and the range keeps at least two frames, because a run covering only the
     * anchor has nothing to propagate.
     */
    const setPropRange = useCallback(
        (first: number, last: number) => {
            let clampedFirst = Math.max(0, Math.min(first, propAnchor));
            let clampedLast = Math.min(
                clip.frameCount - 1,
                Math.max(last, propAnchor),
            );
            if (clampedLast - clampedFirst < 1) {
                if (clampedLast < clip.frameCount - 1)
                    clampedLast = clampedFirst + 1;
                else clampedFirst = Math.max(0, clampedLast - 1);
            }
            setPropFirst(clampedFirst);
            setPropLast(clampedLast);
        },
        [propAnchor, clip.frameCount],
    );

    /**
     * Draw the run as it arrives.
     *
     * Frames the poll loop has received wait here and are painted one per frame
     * interval, so a burst of masks plays back instead of snapping to its end. The
     * clock keeps running while a job is live even with the queue empty — the next
     * batch resumes playback — and drains the remainder once the run settles.
     */
    useEffect(() => {
        if (!propagating && propPlaybackRef.current.length === 0) return;
        const fps = clip.fps > 0 ? clip.fps : 10;
        const interval = Math.max(33, Math.round(1000 / Math.min(fps, 30)));
        let raf = 0;
        let last = performance.now();
        let acc = 0;
        const step = (now: number) => {
            acc += now - last;
            last = now;
            const queue = propPlaybackRef.current;
            while (acc >= interval && queue.length > 0) {
                acc -= interval;
                const frame = queue.shift();
                if (frame !== undefined) setFrameIndex(frame);
            }
            if (queue.length === 0 && !propagating) return;
            raf = requestAnimationFrame(step);
        };
        raf = requestAnimationFrame(step);
        return () => cancelAnimationFrame(raf);
    }, [propagating, clip.fps]);

    /**
     * The selected object's frame states, for the timeline strip.
     *
     * Two marks and a background: a frame a human drew or corrected is `human`
     * (the object's own colour), a frame the tracker produced is `propagated`
     * (orange), and the strip's background is every frame with no mask. A run under
     * review counts as propagated — it is machine output until its object is left.
     */
    const timelineStates = useMemo<TimelineFrameState[]>(() => {
        const count = clip.frameCount;
        const states: TimelineFrameState[] = new Array(count).fill("none");
        if (!selected) return states;
        for (let frame = 0; frame < count; frame++) {
            if (clip.rawMaskAt(selected, frame) === null) continue;
            states[frame] = verifiedFrames.has(frameKey(selected.id, frame))
                ? "human"
                : "propagated";
        }
        // The run under review counts as propagated — but only on its own
        // object's timeline. A run that belongs to a different object (a batch
        // member, or one whose object has been deleted) must never paint this
        // object's frames.
        if (propRun && selected && propRun.trackletId === selected.id) {
            for (const mask of propRun.masks.values()) {
                if (mask.area <= 0) continue;
                if (mask.frameIndex < 0 || mask.frameIndex >= count) continue;
                if (states[mask.frameIndex] === "none")
                    states[mask.frameIndex] = "propagated";
            }
        }
        return states;
    }, [clip, selected, verifiedFrames, propRun]);

    /**
     * What a run would cover: the Shift-clicked batch, or just the selected
     * object so one-click propagation keeps working.
     */
    const propTargets = useMemo(
        () =>
            propBatch.length > 0 ? propBatch : selected ? [selected.id] : [],
        [propBatch, selected],
    );
    /**
     * The targets that can actually seed a run, i.e. have a mask on this frame.
     *
     * A batch member without one is reported rather than fatal: dropping it is
     * better than refusing to propagate the objects that are ready.
     */
    const propReady = useMemo(
        () =>
            propTargets.filter((id) => {
                const tracklet = clip.tracklets.find((item) => item.id === id);
                return tracklet
                    ? humanMaskAt(tracklet, frameIndex) !== null
                    : false;
            }),
        [propTargets, clip, frameIndex, humanMaskAt],
    );

    /**
     * Whether this frame can seed anything at all.
     *
     * A run is conditioned on human input, so a frame nothing verified covers is
     * not an anchor. The action stays on the bar when that happens rather than
     * disappearing: it is greyed out, and pressing it says why.
     */
    const propagateBlocked = !propagating && propReady.length === 0;
    /**
     * Whether the action is unavailable for a reason the frame cannot fix.
     *
     * A stopped tracker or a range of one frame is a dead end the bar already
     * states outright, so the button is disabled rather than click-to-explain.
     */
    const propagateLocked =
        propagating ||
        (!propagateBlocked &&
            (!propagateAvailable || propRange.last - propRange.first < 1));

    /**
     * Write finished runs into the clip.
     *
     * Nothing is ever left waiting for an Accept: a run lands the moment it
     * finishes, so leaving the tool cannot be the only way to keep the masks. A
     * frame's provenance is untouched — what the run wrote still reads as
     * `propagated` on the timeline until a person verifies it, and a frame a
     * person verified was never the run's to write.
     */
    const commitRuns = useCallback(
        (runs: WritableRun[], partial: boolean) => {
            if (runs.length === 0) return;
            const found = runs.reduce(
                (total, run) =>
                    total +
                    [...run.masks.values()].filter((mask) => mask.area > 0)
                        .length,
                0,
            );
            if (found > 0) {
                setClip((prev) => writeRuns(prev, runs, isVerified, partial));
                refresh();
                setSaveNotice({
                    kind: partial ? "info" : "success",
                    text: partial
                        ? `Stopped — kept ${found} propagated frame${found === 1 ? "" : "s"}.`
                        : `Propagated ${found} frame${found === 1 ? "" : "s"}.`,
                });
            }
            // The runs are in the clip now, so nothing is under review: the bar
            // goes back to its pre-run form, ready for the next one.
            propPlaybackRef.current = [];
            propPlaybackJobRef.current = null;
            setPropRun(null);
            setPropQueue([]);
        },
        [isVerified, refresh],
    );

    /**
     * Queue a run for every target, then follow them as they go.
     *
     * Each object is its own job and the backend works through them one at a
     * time, so queueing ten objects costs the same GPU memory as queueing one.
     * The reviewer streams whichever run is going; the queue keeps every job's
     * own copy of its frames, so switching rows costs nothing.
     */
    const runPropagation = useCallback(async () => {
        if (propagating) return;
        const sessionId =
            frames instanceof SessionFrameSource ? frames.sessionId : null;
        if (!sessionId) {
            setPropError("This clip is not open in a backend session.");
            return;
        }
        const anchor = frameIndex;
        const first = propRange.first;
        const last = propRange.last;
        // The range itself says which way to walk: one that starts at the anchor
        // is forward-only, one that ends there is backward-only, otherwise both.
        const direction = "both";
        if (last - first < 1) return;
        const targets = propReady;
        if (targets.length === 0) {
            setPropError(
                `Nothing to propagate: no selected ${vocab.unit} has a mask on frame ${anchor + 1}.`,
            );
            return;
        }
        const skipped = propTargets.filter((id) => !propReady.includes(id));

        const controller = new AbortController();
        propAbortRef.current = controller;
        // The run drives the frame itself, so clip playback would fight it.
        setPlaying(false);
        setPropagating(true);
        setPropError(
            skipped.length === 0
                ? null
                : `Skipped ${skipped.length} ${skipped.length === 1 ? vocab.unit : vocab.units} with no mask on frame ${anchor + 1}.`,
        );
        setPropRun(null);
        setPropQueue([]);
        propPlaybackRef.current = [];
        propPlaybackJobRef.current = null;
        // Declared outside the `try` so the `finally` can land whatever the run
        // produced in the clip, however it ends.
        const entries: PropQueueEntry[] = [];
        try {
            // Queue every object up front. The queue lives on the backend, so it
            // keeps working through the batch even if this panel is closed.
            for (const id of targets) {
                const tracklet = clip.tracklets.find((item) => item.id === id);
                const mask = tracklet ? clip.rawMaskAt(tracklet, anchor) : null;
                if (!tracklet || !mask) continue;
                // Seed the run with every mask a person has verified in the range:
                // that is what makes a re-run continue from the corrections.
                const pins = pinsFor(id, first, last, anchor);
                const job = await startPropagation({
                    sessionId,
                    anchorFrame: anchor,
                    mask,
                    direction,
                    first,
                    last,
                    objectId: tracklet.id,
                    pins: pins.size > 0 ? pins : undefined,
                    signal: controller.signal,
                });
                entries.push({
                    jobId: job.jobId,
                    trackletId: tracklet.id,
                    label: tracklet.label,
                    color: tracklet.color,
                    anchor,
                    first: job.first,
                    last: job.last,
                    state: job.state,
                    framesDone: 0,
                    framesTotal: job.progress.framesTotal,
                    elapsedMs: 0,
                    error: null,
                    writePolicy: "replace-range",
                    pinned: pins.size,
                    windows: [],
                    masks: new Map(),
                });
            }
            if (entries.length === 0) return;
            propJobRef.current = entries.map((entry) => entry.jobId);
            setPropQueue([...entries]);
            // The frames already held per job, as its lowest and highest index:
            // propagation walks outward from the anchor, so a poll asks for what
            // is newer than the top *and* older than the bottom.
            const cursor = new Map<string, { min: number; max: number }>();
            let watching = 0;
            let live = entries.map((entry) => entry.jobId);
            while (live.length > 0) {
                // Poll the whole batch in one pass. Jobs that are still queued
                // answer immediately with their position, so this stays cheap
                // even with a long queue.
                const states = await Promise.all(
                    live.map((jobId) => {
                        const held = cursor.get(jobId);
                        return getPropagationJob(jobId, {
                            since: held?.max ?? -1,
                            until: held?.min,
                            signal: controller.signal,
                        });
                    }),
                );
                const arrived = new Map<string, number[]>();
                for (const state of states) {
                    const entry = entries.find(
                        (item) => item.jobId === state.jobId,
                    );
                    if (!entry) continue;
                    const batch: number[] = [];
                    for (const item of state.masks) {
                        entry.masks.set(item.frameIndex, item);
                        batch.push(item.frameIndex);
                        const held = cursor.get(state.jobId);
                        if (held) {
                            held.min = Math.min(held.min, item.frameIndex);
                            held.max = Math.max(held.max, item.frameIndex);
                        } else {
                            cursor.set(state.jobId, {
                                min: item.frameIndex,
                                max: item.frameIndex,
                            });
                        }
                    }
                    arrived.set(state.jobId, batch);
                    entry.state = state.state;
                    entry.framesDone = state.progress.framesDone;
                    entry.framesTotal = state.progress.framesTotal;
                    entry.elapsedMs = Number(state.plan?.elapsed_ms ?? 0);
                    entry.error = state.error;
                    // The run's windows only exist once the worker has planned it.
                    entry.windows = (state.plan?.windows ?? [])
                        .map((window) => window.start)
                        .filter((frame) => frame > 0);
                }
                live = entries.filter(isLive).map((entry) => entry.jobId);
                // Keep following a run that is still going, so its masks keep
                // streaming in; once they have all settled, stay where we are
                // so the last object's frames stay on screen for review.
                if (!isLive(entries[watching])) {
                    const next = entries.findIndex(isLive);
                    if (next >= 0) watching = next;
                }
                // Follow whichever run the queue is working on.
                const shown = entries[watching];
                // Hand this run's new frames to the frame clock. Switching rows
                // drops the previous run's queue so a new row starts clean.
                if (shown.jobId !== propPlaybackJobRef.current) {
                    propPlaybackJobRef.current = shown.jobId;
                    propPlaybackRef.current = [];
                }
                const newFrames = arrived.get(shown.jobId);
                if (newFrames && newFrames.length > 0) {
                    const queue = propPlaybackRef.current;
                    queue.push(...newFrames);
                    // A run that outpaces playback must not build an unbounded
                    // backlog: drop the oldest frames and catch up to the front.
                    if (queue.length > PROP_PLAYBACK_MAX) {
                        queue.splice(0, queue.length - PROP_PLAYBACK_MAX);
                    }
                }
                setPropQueue([...entries]);
                setPropRun(entryToRun(shown, propStatus));
                if (live.length === 0) break;
                await sleep(PROPAGATE_POLL_MS, controller.signal);
            }
            const failed = entries.filter((entry) => entry.state === "failed");
            if (failed.length > 0) {
                // One line per failure, and the others are still reviewable.
                setPropError(
                    failed
                        .map(
                            (entry) =>
                                `${entry.label} #${entry.trackletId}: ${entry.error ?? "the run failed"}`,
                        )
                        .join(" · "),
                );
            }
            // The loop above already left the canvas on the last frame the run
            // produced, so there is nothing to move to here: it only reports the
            // runs that failed.
        } catch (cause) {
            if (controller.signal.aborted) return;
            setPropError(
                cause instanceof Error ? cause.message : String(cause),
            );
        } finally {
            if (propAbortRef.current === controller) {
                propAbortRef.current = null;
                propJobRef.current = [];
                setPropagating(false);
                // The run lands in the clip on its own, whether it finished or
                // was stopped: a stopped one is written additively, so the
                // frames it never reached keep what they had.
                commitRuns(entries, controller.signal.aborted);
            }
        }
    }, [
        propagating,
        propReady,
        propTargets,
        propRange,
        frameIndex,
        frames,
        clip,
        propStatus,
        vocab.unit,
        vocab.units,
        pinsFor,
        commitRuns,
    ]);

    /** Queue or unqueue a tracklet: Shift-click in the tracklet list. */
    const togglePropBatch = useCallback((id: number) => {
        setPropBatch((current) =>
            current.includes(id)
                ? current.filter((item) => item !== id)
                : [...current, id],
        );
    }, []);

    const cancelRunningPropagation = useCallback(() => {
        // Stop the whole queue, not just the run being watched: that is what Stop
        // means once several objects have been queued.
        for (const jobId of propJobRef.current) {
            void cancelPropagation(jobId).catch(() => undefined);
        }
        propJobRef.current = [];
        // The abort is left to clear the ref: the run's `finally` commits what it
        // already produced, and it only does that while it still owns the ref.
        propAbortRef.current?.abort();
        setPropagating(false);
        setPropError(null);
    }, []);

    /**
     * The bar's one action.
     *
     * A frame that cannot anchor a run does not hide the button — the reviewer
     * pressed it, so it answers: a transient hint beside it says what is wrong
     * with this frame. A frame that can start a run simply starts one.
     */
    const onPropagateClick = useCallback(() => {
        if (propagating) return;
        if (propagateBlocked) {
            const rect = propButtonRef.current?.getBoundingClientRect();
            setPropHint(
                rect
                    ? {
                          x: rect.right,
                          y: rect.bottom + 6,
                          frame: frameIndex,
                          tool,
                      }
                    : null,
            );
            return;
        }
        void runPropagation();
    }, [propagating, propagateBlocked, runPropagation, frameIndex, tool]);

    const propPreview =
        tool === "propagate" ? (propRun?.masks.get(frameIndex) ?? null) : null;
    /**
     * The colour a propagated preview is drawn in: the object's own label colour.
     *
     * The video shows a mask the same way whether a person drew it or the tracker
     * did; the orange is the timeline's job, not the canvas's.
     */
    const propPreviewColor = useMemo(
        () =>
            propRun
                ? clip.tracklets.find((t) => t.id === propRun.trackletId)?.color
                : undefined,
        [propRun, clip.tracklets],
    );
    /**
     * What the run produced, split into what Accept will write and what it leaves.
     *
     * A run replaces the range it covers, but a frame a human has verified is never
     * overwritten — including one they *cleared*, because clearing is the statement
     * that the object is not there.
     */
    const propSummary = useMemo(() => {
        if (!propRun) return null;
        const found = [...propRun.masks.values()].filter((m) => m.area > 0);
        const accepted = found.filter(
            (m) => !isVerified(propRun.trackletId, m.frameIndex),
        );
        return { found: found.length, accepted };
    }, [propRun, isVerified]);

    const stepPropagated = useCallback(
        (delta: 1 | -1) => {
            if (!propRun) return;
            const span = propRun.last - propRun.first + 1;
            let next = frameIndex;
            for (let k = 0; k < span; k++) {
                next = next + delta;
                if (next < propRun.first) next = propRun.last;
                if (next > propRun.last) next = propRun.first;
                if (propRun.masks.has(next)) break;
            }
            setFrameIndex(next);
        },
        [propRun, frameIndex],
    );

    /**
     * Commit the batch's runs into the clip.
     *
     * Every run lands at once, because a run is written as it settles: this is
     * the gesture for the gap between the last frame arriving and that write,
     * and the safety net for leaving the tool before it happens.
     */
    const acceptPropagation = useCallback(() => {
        // Every run in the batch lands at once, so this is the explicit gesture
        // for the moment between a run finishing and its write, and the safety
        // net for leaving the tool before that write happens.
        const runs: WritableRun[] =
            propQueue.length > 0 ? propQueue : propRun ? [propRun] : [];
        commitRuns(runs, false);
    }, [propQueue, propRun, commitRuns]);

    //: Assigned here, below `acceptPropagation`; `changeTool` commits a finished
    //: run through it, so leaving the tool keeps the run instead of discarding it.
    acceptPropagationRef.current = acceptPropagation;

    useEffect(() => {
        if (!playing) return;
        let raf = 0;
        let last = performance.now();
        let accumulator = 0;
        const fps = clip.fps;
        const count = clip.frameCount;

        const step = (now: number) => {
            const elapsed = (now - last) / 1000;
            last = now;
            accumulator += elapsed * fps;
            if (accumulator >= 1) {
                const advance = Math.min(Math.floor(accumulator), count);
                accumulator -= Math.floor(accumulator);
                setFrameIndex((index) => (index + advance) % count);
            }
            raf = requestAnimationFrame(step);
        };

        raf = requestAnimationFrame(step);
        return () => cancelAnimationFrame(raf);
    }, [playing, clip.fps, clip.frameCount]);

    // The whole keyboard map lives in a hook; everything it branches on is
    // passed in, so the component keeps only the wiring.
    useWorkspaceShortcuts(
        {
            exportOpen,
            tool,
            method,
            polygonLength: polygon.length,
            hasSelection: selected !== null,
            selectedHasHumanMaskHere,
            hasPropRun: propRun !== null,
            propagating,
        },
        {
            stepFrame,
            changeTool,
            changeMethod,
            closePolygon,
            commitMask,
            undo,
            acceptPropagation,
            runPropagation,
            setPlaying,
            setBrushSize,
        },
    );

    /**
     * Ask once before uncommitted work is thrown away.
     *
     * The spec's §6.2 leave-guard: exporting or closing with drafts still open
     * prompts, so navigating away cannot silently lose them.
     */
    const confirmDraftsDiscarded = useCallback(() => {
        const count = drafts.size;
        if (count === 0) return true;
        return window.confirm(
            `${count} frame${count === 1 ? " has" : "s have"} uncommitted ${vocab.unit} mask draft${count === 1 ? "" : "s"}. Discard ${count === 1 ? "it" : "them"} and continue?`,
        );
    }, [drafts.size, vocab.unit]);

    /**
     * Save the annotated project.
     *
     * The committed state is packed back into a `.project` archive and handed to
     * the browser, so the review can be resumed later. Uncommitted drafts would
     * not be in the file, so they are guarded exactly like export.
     */
    const handleSave = useCallback(async () => {
        if (!confirmDraftsDiscarded()) return;
        const result = await saveProject();
        if (result.ok) setSavedEditCount(clip.editCount);
        setSaveNotice(
            result.ok
                ? { kind: "success", text: `Saved ${result.fileName}.` }
                : { kind: "info", text: result.error },
        );
    }, [clip.editCount, confirmDraftsDiscarded, saveProject]);

    return (
        <div className={styles.workspace}>
            <header className={styles.header}>
                <span className={styles.brand} title="Field Lab">
                    <Icon name="leaf" size={18} />
                </span>

                <div className={styles.identity}>
                    <div className={styles.titleRow}>
                        <h1 className={styles.title}>{clip.name}</h1>
                        <Chip
                            tone={
                                clip.mode === "semantic" ? "accent" : "neutral"
                            }
                            icon="lock"
                            title={`${vocab.title} segmentation — ${vocab.description} Fixed when the project was created.`}
                        >
                            {vocab.title}
                        </Chip>
                    </div>
                    <div className={styles.meta}>
                        <span className="num">
                            {clip.width}×{clip.height}
                        </span>
                        <span className={styles.sep} />
                        <span className="num">{clip.frameCount} frames</span>
                        <span className={styles.sep} />
                        <span className="num">{clip.fps} fps</span>
                        <span className={styles.sep} />
                        <span className="num">
                            {clip.tracklets.length}{" "}
                            {clip.tracklets.length === 1
                                ? vocab.unit
                                : vocab.units}
                        </span>
                    </div>
                </div>

                <div className={styles.spacer} />

                <Button
                    variant="ghost"
                    icon="folder"
                    disabled={propagating}
                    aria-label="Open another project"
                    title={
                        propagating
                            ? "Stop the run before opening another project"
                            : "Open another project"
                    }
                    onClick={() => {
                        if (confirmDraftsDiscarded()) onReset();
                    }}
                />

                <Button
                    variant="primary"
                    icon="save"
                    disabled={saveBusy || propagating || exportBusyId !== null}
                    title={
                        propagating
                            ? "Stop the run before saving — its masks are not saved yet"
                            : saveProgress
                              ? `Saving ${saveProgress}…`
                              : "Download the annotated project as a .project archive"
                    }
                    onClick={() => void handleSave()}
                >
                    {saveBusy ? "Saving…" : "Save"}
                </Button>

                <Button
                    variant="default"
                    icon="download"
                    disabled={exportBusyId !== null || propagating || saveBusy}
                    title={
                        propagating
                            ? "Stop the run before exporting — its masks are not saved yet"
                            : undefined
                    }
                    onClick={() => {
                        // Leaving with drafts open prompts once, before
                        // anything is written.
                        if (!confirmDraftsDiscarded()) return;
                        openExport();
                    }}
                >
                    {exportBusyId !== null ? "Exporting…" : "Export"}
                </Button>
            </header>

            {notice && (
                <div
                    className={`${styles.notice} ${
                        notice.kind === "success" ? styles.noticeSuccess : ""
                    }`}
                    role="status"
                >
                    <span>{notice.text}</span>
                    <button
                        type="button"
                        className={styles.noticeClose}
                        onClick={dismissNotice}
                        aria-label="Dismiss"
                    >
                        ×
                    </button>
                </div>
            )}

            <div className={styles.body} ref={bodyRef} style={bodyStyle}>
                <Toolbar
                    mode={clip.mode}
                    tool={tool}
                    sam={sam}
                    modelName={modelName}
                    canEdit={selected !== null}
                    canPropagate={selectedHasHumanMaskHere}
                    propagateModel={propagateModel}
                    busy={propagating}
                    onToolChange={changeTool}
                    onRefreshStatus={refreshSam}
                />

                <section className={styles.videoCol}>
                    {/* Select has no controls, but every other tool draws a bar
                        here. Keep the empty row so switching tools never shifts
                        the stage up or down. */}
                    {tool === "review" && (
                        <div
                            className={styles.promptBar}
                            aria-hidden="true"
                            data-placeholder="true"
                        />
                    )}
                    {tool === "propagate" && selected && (
                        <div className={styles.promptBar} role="toolbar">
                            <span className={styles.promptTitle}>
                                Propagate · {selected.label} #{selected.id} ·
                                from frame {propAnchor + 1}
                            </span>
                            {propRun ? (
                                <span className={styles.promptStatus}>
                                    {propagating
                                        ? `Propagating · ${propRun.framesDone}/${propRun.framesTotal} frames`
                                        : `${propSummary?.found ?? 0} frame${(propSummary?.found ?? 0) === 1 ? "" : "s"} propagated`}
                                </span>
                            ) : (
                                <>
                                    <span className={styles.promptStatus}>
                                        Frames {propRange.first + 1}–
                                        {propRange.last + 1} · drag the bars on
                                        the timeline to set the range
                                    </span>
                                    {propagateNote && (
                                        <span className={styles.promptError}>
                                            {propagateNote}
                                        </span>
                                    )}
                                    {propError && (
                                        <span className={styles.promptError}>
                                            {propError}
                                        </span>
                                    )}
                                </>
                            )}
                            <span className={styles.spacer} />
                            {propagating ? (
                                <button
                                    type="button"
                                    className="btn"
                                    onClick={cancelRunningPropagation}
                                    title="Stop the run; frames already produced are kept"
                                >
                                    Stop
                                </button>
                            ) : (
                                !propRun && (
                                    <button
                                        type="button"
                                        className="btn"
                                        onClick={() => changeTool("review")}
                                        title="Esc"
                                    >
                                        Cancel
                                    </button>
                                )
                            )}
                            {/* The one action, present in every state: a frame
                                that cannot seed a run greys it out rather than
                                hiding it, so pressing it can say why. */}
                            <button
                                type="button"
                                ref={propButtonRef}
                                className={`btn btnPrimary ${
                                    propagateBlocked ? styles.actionBlocked : ""
                                }`}
                                disabled={propagateLocked}
                                onClick={onPropagateClick}
                                title={
                                    propagating
                                        ? "Stop the run first"
                                        : propagateBlocked
                                          ? "A run starts from a frame you drew or corrected"
                                          : propReady.length > 1
                                            ? `Queue one run per object (${propReady.length})`
                                            : "Enter"
                                }
                            >
                                {propagating
                                    ? "Propagating…"
                                    : propReady.length > 1
                                      ? `Propagate ${propReady.length} objects`
                                      : "Propagate"}
                            </button>
                            {propHint &&
                                propHint.frame === frameIndex &&
                                propHint.tool === tool && (
                                    <div
                                        className={styles.actionHint}
                                        style={{
                                            left: propHint.x,
                                            top: propHint.y,
                                        }}
                                        role="status"
                                    >
                                        Start at a frame you drew or corrected:
                                        a run is anchored on human input.
                                    </div>
                                )}
                        </div>
                    )}
                    {(tool === "addMask" || tool === "editMask") && (
                        <div className={styles.promptBar} role="toolbar">
                            <span className={styles.promptTitle}>
                                {tool === "addMask"
                                    ? "Add mask"
                                    : `Edit mask · ${selected?.label ?? ""} #${selectedId}`}
                            </span>

                            <div className={styles.barDivider} />

                            <div
                                className="segmented"
                                role="radiogroup"
                                aria-label="Drawing method"
                            >
                                {availableMethods.map((value) => (
                                    <button
                                        key={value}
                                        type="button"
                                        role="radio"
                                        aria-checked={method === value}
                                        className={`segment ${method === value ? "segmentActive" : ""}`}
                                        onClick={() => changeMethod(value)}
                                        title={
                                            value === "point"
                                                ? `Click the object to segment it with ${modelName}`
                                                : value === "box"
                                                  ? `Drag a box around the object (${modelName})`
                                                  : value === "text"
                                                    ? `Type a class name; ${modelName} finds every match`
                                                    : value === "polygon"
                                                      ? "Click vertices, close with Enter, double-click or right-click (P)"
                                                      : "Drag to paint; Shift/right-drag erases; [ ] resize (B)"
                                        }
                                    >
                                        {METHOD_LABELS[value]}
                                    </button>
                                ))}
                            </div>

                            {box && (
                                <button
                                    type="button"
                                    className="btn"
                                    onClick={() => {
                                        // Drop the box and re-run with the clicks
                                        // alone, so the result matches the prompt
                                        // the reviewer is now looking at.
                                        setBox(null);
                                        setPromptError(null);
                                        if (prompt.length > 0)
                                            segmentPoints(prompt);
                                        else setCandidate(null);
                                    }}
                                    title="The box is part of the current prompt. Clear it to prompt with clicks alone."
                                >
                                    Clear box prompt
                                </button>
                            )}

                            {(method === "brush" || method === "polygon") && (
                                <div
                                    className="segmented"
                                    role="radiogroup"
                                    aria-label="Paint mode"
                                >
                                    {(["add", "erase"] as PaintMode[]).map(
                                        (value) => (
                                            <button
                                                key={value}
                                                type="button"
                                                role="radio"
                                                aria-checked={
                                                    paintMode === value
                                                }
                                                className={`segment ${paintMode === value ? "segmentActive" : ""}`}
                                                onClick={() =>
                                                    setPaintMode(value)
                                                }
                                                title={
                                                    value === "add"
                                                        ? "New strokes and shapes are added to the draft"
                                                        : "New strokes and shapes are subtracted from the draft"
                                                }
                                            >
                                                {value === "add"
                                                    ? "Add"
                                                    : "Erase"}
                                            </button>
                                        ),
                                    )}
                                </div>
                            )}

                            <div className={styles.barDivider} />

                            {/* Deleting the committed mask is an Edit-mask act: in Add
                                mask the draft is a brand new object, so clearing the
                                selected one has nothing to do with what is being
                                drawn. Whole-object and frame deletion live on the
                                object list and the tool rail. */}
                            {tool === "editMask" &&
                                selected &&
                                selectedHasMaskHere && (
                                    <button
                                        type="button"
                                        className="btn"
                                        onClick={() => {
                                            if (!selected) return;
                                            if (
                                                selected.maskFrames.count <= 1
                                            ) {
                                                // The object's only mask: removing
                                                // it deletes the object, so confirm
                                                // first and land in Add mask to draw
                                                // it again.
                                                setPendingDelete({
                                                    id: selected.id,
                                                    then: "addMask",
                                                });
                                                return;
                                            }
                                            clearMaskOnFrame();
                                        }}
                                        title={
                                            selected.maskFrames.count <= 1
                                                ? "This is the object's only mask, so deleting it removes the object. You will land in Add mask to draw it again."
                                                : "Delete the mask on this frame and draw it again. A redraw then replaces it, instead of being added to the mask you are rejecting."
                                        }
                                    >
                                        Delete mask on this frame
                                    </button>
                                )}

                            {method === "brush" && (
                                <label
                                    className={styles.brushSize}
                                    title="Brush diameter in frame pixels ([ and ])"
                                >
                                    Size
                                    <input
                                        type="range"
                                        min={1}
                                        max={200}
                                        step={1}
                                        value={Math.min(200, brushSize)}
                                        onChange={(event) =>
                                            setBrushSize(
                                                Number(event.target.value),
                                            )
                                        }
                                    />
                                    <span className={styles.brushValue}>
                                        {brushSize}px
                                    </span>
                                </label>
                            )}

                            {!semantic && tool === "addMask" && (
                                <select
                                    className={styles.promptSelect}
                                    value={
                                        clip.labelById(newLabelId)
                                            ? String(newLabelId)
                                            : ""
                                    }
                                    onChange={(event) =>
                                        setNewLabelId(
                                            event.target.value === ""
                                                ? null
                                                : Number(event.target.value),
                                        )
                                    }
                                    aria-label="Label for the new object"
                                    title="Label the object you are about to create. Leave as Unlabelled to label it later."
                                >
                                    <option value="">Unlabelled</option>
                                    {clip.labels.map((label) => (
                                        <option key={label.id} value={label.id}>
                                            {label.name}
                                        </option>
                                    ))}
                                </select>
                            )}

                            {/* The name box doubles as the text-prompt box: in
                                semantic mode the class name is required for
                                every method, but in instance mode it is only
                                needed for the Text method, which finds a
                                concept rather than naming a class. */}
                            {((semantic && tool === "addMask") ||
                                (!semantic && method === "text")) && (
                                <>
                                    <input
                                        className={styles.promptInput}
                                        list="vsr-class-names"
                                        placeholder={
                                            semantic
                                                ? "class name (e.g. coral)"
                                                : "text prompt (e.g. shark)"
                                        }
                                        value={className}
                                        onChange={(event) =>
                                            setClassName(event.target.value)
                                        }
                                        onKeyDown={(event) => {
                                            if (event.key === "Enter") {
                                                event.preventDefault();
                                                if (
                                                    method === "point" ||
                                                    method === "box" ||
                                                    method === "text"
                                                )
                                                    runTextPrompt();
                                                else commitMask();
                                            } else if (event.key === "Escape") {
                                                event.preventDefault();
                                                changeTool("review");
                                            }
                                        }}
                                        aria-label={
                                            semantic
                                                ? "Class name"
                                                : "Text prompt"
                                        }
                                        title={
                                            method === "point" ||
                                            method === "box" ||
                                            method === "text"
                                                ? semantic
                                                    ? "Sent to SAM 3 as the concept to find. Matches an existing class by name, otherwise a new class is created. Enter runs the model."
                                                    : "Sent to SAM 3 as the concept to find. It names the objects you add; the label picker overrides it. Enter runs the model."
                                                : "Class the drawn mask is added to (matched by name; otherwise a new class is created)."
                                        }
                                    />
                                    <datalist id="vsr-class-names">
                                        {classLabels.map((label) => (
                                            <option key={label} value={label} />
                                        ))}
                                    </datalist>
                                    {(method === "point" ||
                                        method === "box" ||
                                        method === "text") && (
                                        <button
                                            type="button"
                                            className="btn"
                                            disabled={
                                                segmenting ||
                                                (prompt.length === 0 &&
                                                    !className.trim())
                                            }
                                            onClick={runTextPrompt}
                                            title={
                                                semantic
                                                    ? "Run SAM 3 with the current clicks and class name (Enter in the name box)"
                                                    : "Run SAM 3 with the text prompt (Enter in the box)"
                                            }
                                        >
                                            Find
                                        </button>
                                    )}
                                </>
                            )}

                            <span className={styles.promptStatus}>
                                {promptError ? (
                                    <span className={styles.promptError}>
                                        {promptError}
                                    </span>
                                ) : segmenting ? (
                                    "Segmenting…"
                                ) : candidate &&
                                  (method === "point" ||
                                      method === "box" ||
                                      method === "text") ? (
                                    <>
                                        {candidate.area === 0 ? (
                                            <span
                                                className={styles.promptError}
                                            >
                                                Nothing found.
                                            </span>
                                        ) : (
                                            <>
                                                {isTextSplit
                                                    ? `${candidate.instances.length} objects · `
                                                    : ""}
                                                {candidate.area.toLocaleString()}{" "}
                                                px
                                                {draft ? " · in draft" : ""}
                                                {isTextSplit && (
                                                    <>
                                                        <button
                                                            type="button"
                                                            className="btn btnPrimary"
                                                            onClick={() =>
                                                                commitInstances(
                                                                    "all",
                                                                )
                                                            }
                                                            title={`Create one ${vocab.unit} per detected object`}
                                                        >
                                                            Add all{" "}
                                                            {
                                                                candidate
                                                                    .instances
                                                                    .length
                                                            }{" "}
                                                            as separate{" "}
                                                            {vocab.unit}s
                                                        </button>
                                                        <button
                                                            type="button"
                                                            className="btn"
                                                            onClick={() =>
                                                                commitInstances(
                                                                    "largest",
                                                                )
                                                            }
                                                            title={`Create a single ${vocab.unit} from the largest object`}
                                                        >
                                                            Largest only
                                                        </button>
                                                    </>
                                                )}
                                            </>
                                        )}
                                    </>
                                ) : draft ? (
                                    `Draft ${draftArea.toLocaleString()} px`
                                ) : tool === "editMask" ? (
                                    originalMask ? (
                                        "Erased — save to drop the mask."
                                    ) : (
                                        "No mask on this frame."
                                    )
                                ) : method === "text" ? (
                                    semantic ? (
                                        "Type the class name, then press Enter."
                                    ) : (
                                        "Type a concept, then press Enter — SAM finds every match."
                                    )
                                ) : method === "point" || method === "box" ? (
                                    semantic ? (
                                        "Click the class, or type its name."
                                    ) : (
                                        "Click the object · Shift-click to exclude."
                                    )
                                ) : method === "polygon" ? (
                                    "Click to add points · Enter closes."
                                ) : (
                                    "Drag to paint · Shift-drag erases."
                                )}
                            </span>
                            <div className={styles.spacer} />
                            {method === "polygon" && polygon.length >= 3 && (
                                <button
                                    type="button"
                                    className="btn"
                                    onClick={closePolygon}
                                    title="Enter"
                                >
                                    Close polygon
                                </button>
                            )}
                            {/*
                                One recovery button, not three. Undo, Reset and
                                Cancel all read as "throw something away" and sat
                                side by side, so the bar now offers a single Clear
                                that drops every pending mask on this frame. Undo
                                (Backspace) and leaving the tool (Esc) still work
                                from the keyboard, where they cost no clutter.
                            */}
                            <button
                                type="button"
                                className="btn"
                                disabled={!canUndo && !draftChanged}
                                onClick={discardDraft}
                                title="Throw away every pending mask on this frame — the draft, the clicks and any polygon — and start over. Esc leaves the tool without discarding."
                            >
                                Clear
                            </button>
                            <button
                                type="button"
                                className="btn btnPrimary"
                                disabled={
                                    segmenting ||
                                    (tool === "editMask"
                                        ? !draftChanged
                                        : !finalMask)
                                }
                                onClick={commitMask}
                                title="Enter"
                            >
                                {tool === "editMask"
                                    ? finalMask
                                        ? "Save mask"
                                        : "Remove mask on this frame"
                                    : !semantic
                                      ? `Add as new ${vocab.unit}`
                                      : targetClass
                                        ? `Add to "${targetClass.label}"`
                                        : className.trim()
                                          ? `New class "${className.trim()}"`
                                          : "Add as new class"}
                            </button>
                        </div>
                    )}
                    <TimelineStrip
                        frameCount={clip.frameCount}
                        frameIndex={frameIndex}
                        states={timelineStates}
                        color={selected?.color}
                        label={
                            selected
                                ? `${vocab.unit} #${selected.id} · ${selected.label}`
                                : `No ${vocab.unit} selected`
                        }
                        rangeFirst={
                            tool === "propagate" ? propRange.first : undefined
                        }
                        rangeLast={
                            tool === "propagate" ? propRange.last : undefined
                        }
                        rangeAnchor={
                            tool === "propagate" ? propAnchor : undefined
                        }
                        onRangeChange={
                            tool === "propagate" && !propRun && !propagating
                                ? setPropRange
                                : undefined
                        }
                        onSeek={setFrameIndex}
                    />
                    <VideoPanel
                        clip={clip}
                        frames={frames}
                        frameIndex={frameIndex}
                        playing={playing}
                        selectedTrackletId={selectedId}
                        maskOpacity={maskOpacity}
                        onFrameChange={setFrameIndex}
                        onPlayToggle={togglePlay}
                        onStep={(delta) =>
                            // While reviewing a run, step between the frames it
                            // produced: the anchor has no propagated mask, so a
                            // plain +/-1 step would land on an empty canvas.
                            tool === "propagate" && propRun
                                ? stepPropagated(delta >= 0 ? 1 : -1)
                                : stepFrame(delta)
                        }
                        onMaskOpacityChange={setMaskOpacity}
                        tool={tool}
                        method={method}
                        paintMode={paintMode}
                        brushSize={brushSize}
                        prompt={prompt}
                        box={box}
                        polygon={polygon}
                        outline={outlineRings}
                        draft={draftDecoded}
                        candidateColor={
                            tool === "propagate" ? propPreviewColor : undefined
                        }
                        candidate={
                            tool === "propagate"
                                ? propPreview
                                    ? rleToDecoded(propPreview.rle)
                                    : null
                                : method === "point" ||
                                    method === "box" ||
                                    method === "text"
                                  ? (candidate?.mask ?? null)
                                  : null
                        }
                        editingTrackletId={
                            tool === "editMask"
                                ? selectedId
                                : tool === "propagate" &&
                                    propPreview &&
                                    propPreview.area > 0
                                  ? (propRun?.trackletId ?? null)
                                  : null
                        }
                        onPromptPoint={addPromptPoint}
                        onPromptBox={segmentBox}
                        onPolygonPoint={addPolygonPoint}
                        onPolygonClose={closePolygon}
                        onOutlineChange={commitOutline}
                        onStroke={applyStroke}
                        onSelectTracklet={selectOnCanvas}
                        promptHint={
                            tool === "propagate"
                                ? propRun
                                    ? "Enter accepts · Esc cancels"
                                    : "Enter runs the tracker"
                                : semantic
                                  ? "Click the class to find all of it"
                                  : undefined
                        }
                    />
                </section>

                <Splitter
                    orientation="vertical"
                    label="Resize the video and sidebar panels"
                    onDrag={onPanelDrag}
                    onNudge={onPanelNudge}
                    onReset={resetPanelW}
                />

                <aside
                    className={styles.sidebar}
                    ref={sidebarRef}
                    style={sidebarStyle}
                >
                    <TrackletList
                        clip={clip}
                        selectedId={selectedId}
                        onSelect={selectTracklet}
                        batch={propBatch}
                        onToggleBatch={
                            tool === "propagate" ? togglePropBatch : undefined
                        }
                        onDelete={(id) =>
                            setPendingDelete({ id, then: "review" })
                        }
                        onAssign={semantic ? undefined : assignTrackletLabel}
                        onNewLabel={
                            semantic ? undefined : createLabelForTracklet
                        }
                    />
                    {!semantic && (
                        <>
                            <Splitter
                                orientation="horizontal"
                                label="Resize the label list"
                                onDrag={onLabelsDrag}
                                onNudge={onLabelsNudge}
                                onReset={resetLabelsH}
                            />
                            <LabelsPanel
                                clip={clip}
                                selectedLabelId={selectedLabel?.id ?? null}
                                onSelect={selectLabel}
                                onCreate={() => openLabelEditor(null)}
                                onEdit={(labelId) => openLabelEditor(labelId)}
                                onDelete={setPendingLabelDelete}
                            />
                        </>
                    )}
                </aside>
            </div>

            {exportOpen && (
                <ExportMenu
                    title={`Export ${clip.name}`}
                    options={exportOptions}
                    busyId={exportBusyId}
                    busyText={exportProgress}
                    error={exportError}
                    onClose={closeExport}
                />
            )}

            {pendingDelete !== null && (
                <Dialog
                    title={`Delete ${vocab.unit} #${pendingDelete.id}?`}
                    onClose={() => setPendingDelete(null)}
                    footer={
                        <>
                            <Button
                                variant="ghost"
                                onClick={() => setPendingDelete(null)}
                            >
                                Cancel
                            </Button>
                            <Button
                                variant="danger"
                                onClick={() => {
                                    const { id, then } = pendingDelete;
                                    // Only the object Edit mask is attached to
                                    // matters: a row can be deleted while a
                                    // different object is selected.
                                    const wasEditing =
                                        toolRef.current === "editMask" &&
                                        id === selectedId;
                                    deleteTracklet(id);
                                    setPendingDelete(null);
                                    // That object is gone, so Edit mask has
                                    // nothing to correct: land where the
                                    // deletion implies (Add mask to redraw,
                                    // Select for a list deletion). Through
                                    // changeTool so the paint mode and the
                                    // prompt are reset with the tool.
                                    if (wasEditing) changeTool(then);
                                }}
                            >
                                Delete {vocab.unit}
                            </Button>
                        </>
                    }
                >
                    <p>
                        {pendingDeleteTarget
                            ? `“${pendingDeleteTarget.label}” has ${pendingDeleteTarget.maskFrames.count} frame${
                                  pendingDeleteTarget.maskFrames.count === 1
                                      ? ""
                                      : "s"
                              } with a mask. Deleting removes the ${vocab.unit} and every one of them from the clip.`
                            : `This removes the ${vocab.unit} and all of its masks from the clip.`}{" "}
                        This cannot be undone — corrections made elsewhere that
                        reference it are lost too.
                    </p>
                </Dialog>
            )}

            {labelEditor && (
                <LabelEditor
                    label={editorLabel}
                    taxonomy={editorTaxonomy}
                    color={editorColor}
                    onSave={saveLabel}
                    onClose={() => setLabelEditor(null)}
                />
            )}

            {pendingLabelDelete !== null && (
                <Dialog
                    title={`Delete label ${pendingLabelDelete}?`}
                    onClose={() => setPendingLabelDelete(null)}
                    footer={
                        <>
                            <Button
                                variant="ghost"
                                onClick={() => setPendingLabelDelete(null)}
                            >
                                Cancel
                            </Button>
                            <Button
                                variant="danger"
                                onClick={() => deleteLabel(pendingLabelDelete)}
                            >
                                Delete label
                            </Button>
                        </>
                    }
                >
                    <p>
                        {pendingLabel
                            ? `“${pendingLabel.name}” is used by ${pendingLabelCount} ${
                                  pendingLabelCount === 1 ? "object" : "objects"
                              }, which will become unlabelled.`
                            : "This label will be removed."}{" "}
                        This cannot be undone.
                    </p>
                </Dialog>
            )}
        </div>
    );
}
