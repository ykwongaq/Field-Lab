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
    type PropagationDirection,
} from "../lib/propagateApi";
import { usePanelLayout } from "../hooks/usePanelLayout";
import { useExport } from "../hooks/useExport";
import { useWorkspaceShortcuts } from "../hooks/useWorkspaceShortcuts";
import { useLabelEditing } from "../hooks/useLabelEditing";
import { VideoPanel } from "./VideoPanel";
import { PropagationRange } from "./PropagationRange";
import {
    PropagationQueue,
    isLive,
    type PropQueueEntry,
} from "./PropagationQueue";
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

const DEFAULT_PROPAGATE_FORWARD = 10;

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
const PROPAGATE_POLL_MS = 700;

/** The "3/120 frames" read-out for a running propagation. */
function jobProgressText(run: PropagateRun | null): string {
    return run
        ? ` ${run.framesDone}/${run.framesTotal} frames · ${run.masks.size} previewed`
        : "…";
}

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
    const [method, setMethod] = useState<DrawMethod>("point");
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

    const [prompt, setPrompt] = useState<PromptPoint[]>([]);
    const [box, setBox] = useState<PromptBox | null>(null);
    const [candidate, setCandidate] = useState<SegmentResult | null>(null);
    const [segmenting, setSegmenting] = useState(false);
    const [promptError, setPromptError] = useState<string | null>(null);
    const segmentAbortRef = useRef<AbortController | null>(null);
    const [sam, setSam] = useState<Sam3Status | null>(null);

    const [className, setClassName] = useState("");
    /**
     * The object a row's trash button asked to delete, pending confirmation.
     *
     * Removing an object is not undoable, so the list opens the one modal shell
     * rather than deleting on the click.
     */
    const [pendingDelete, setPendingDelete] = useState<number | null>(null);
    const [localNotice, setLocalNotice] = useState<WorkspaceNotice | null>(
        null,
    );
    const notice = localNotice ?? externalNotice;

    // The export chooser's state, runner and per-project options. It is created
    // here, ahead of the keyboard effect that reads `exportOpen`, and its
    // success notice is routed back into the workspace's own notice state.
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
        onExported: (fileName) =>
            setLocalNotice({
                kind: "success",
                text: `Exported ${fileName}.`,
            }),
    });

    const [propStatus, setPropStatus] = useState<PropagateStatus | null>(null);
    // Both directions over the whole clip is the default the reviewer wants:
    // "carry this mask as far as it goes" covers the common case in one click.
    const [propDirection, setPropDirection] =
        useState<PropagationDirection>("both");
    const [propWholeClip, setPropWholeClip] = useState(true);
    const [propBack, setPropBack] = useState(0);
    const [propForward, setPropForward] = useState(DEFAULT_PROPAGATE_FORWARD);
    const [propagating, setPropagating] = useState(false);
    const [propError, setPropError] = useState<string | null>(null);
    const [propRun, setPropRun] = useState<PropagateRun | null>(null);
    const [propSkipExisting, setPropSkipExisting] = useState(true);
    const propAbortRef = useRef<AbortController | null>(null);
    //: Every job of the current batch, so Stop can stop all of them and not just
    //: the one being watched.
    const propJobRef = useRef<string[]>([]);
    //: A queue row picked by hand, which outranks the automatic follow.
    const propWatchRef = useRef<string | null>(null);
    /**
     * A refinement run, prepared but not yet started.
     *
     * The panel's controls are React state, and state written in the same tick is
     * not visible to the run started right after it, so a prepared refinement is
     * handed over through a ref instead. `runPropagation` reads it once and clears
     * it, which keeps the one-click path working without a second code path.
     */
    const propOverrideRef = useRef<{
        target: number;
        anchor: number;
        first: number;
        last: number;
        pins: Map<number, RawRle>;
        writePolicy: "skip-existing" | "replace-range";
    } | null>(null);
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
    /**
     * `${objectId}:${frame}` → the corrected frame that invalidated this mask.
     *
     * Only *derived* frames belong here: correcting frame `k` says nothing about a
     * mask a human drew, and re-propagating is the act of replacing exactly these.
     */
    const [stale, setStale] = useState<Map<string, number>>(() => new Map());
    const dismissNotice = useCallback(() => {
        if (localNotice) setLocalNotice(null);
        else onDismissNotice?.();
    }, [localNotice, onDismissNotice]);

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
        // An uncommitted draft is as worth guarding as a committed edit.
        if (clip.editCount === 0 && drafts.size === 0) return;
        const onBeforeUnload = (event: BeforeUnloadEvent) => {
            event.preventDefault();
        };
        window.addEventListener("beforeunload", onBeforeUnload);
        return () => window.removeEventListener("beforeunload", onBeforeUnload);
    }, [clip.editCount, drafts.size]);

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
        onNotice: (text) => setLocalNotice({ kind: "info", text }),
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
     * Two things follow, and together they are the refinement loop:
     *
     * 1. `k` becomes *verified*, so a later run may anchor on it and must not
     *    overwrite it.
     * 2. Every mask the object holds *after* `k` becomes stale. Those came from a
     *    run that had not seen this correction, and a correction says nothing
     *    about the frames before it, so only the downstream ones are invalidated.
     *
     * Returns the frames it invalidated, so the caller can say how many.
     */
    const markCorrected = useCallback(
        (objectId: number, frame: number) => {
            const tracklet = clip.tracklets.find(
                (item) => item.id === objectId,
            );
            setVerifiedFrames((current) =>
                new Set(current).add(frameKey(objectId, frame)),
            );
            if (!tracklet || tracklet.maskFrames.first < 0) return [];
            const invalidated: number[] = [];
            for (
                let other = Math.max(frame + 1, tracklet.maskFrames.first);
                other <= tracklet.maskFrames.last;
                other++
            ) {
                if (clip.rawMaskAt(tracklet, other) === null) continue;
                if (isVerified(objectId, other)) continue;
                invalidated.push(other);
            }
            if (invalidated.length > 0) {
                setStale((current) => {
                    const next = new Map(current);
                    for (const other of invalidated)
                        next.set(frameKey(objectId, other), frame);
                    return next;
                });
            }
            return invalidated;
        },
        [clip, isVerified],
    );

    /** The selected object's masks that a correction has invalidated, ascending. */
    const staleFrames = useMemo(() => {
        if (!selected) return [];
        const prefix = `${selected.id}:`;
        return [...stale.keys()]
            .filter((key) => key.startsWith(prefix))
            .map((key) => Number(key.slice(prefix.length)))
            .filter((frame) => clip.rawMaskAt(selected, frame) !== null)
            .sort((left, right) => left - right);
    }, [selected, clip, stale]);

    /** The corrected frame those stale frames came from, if any. */
    const refineFrom = useMemo(() => {
        if (!selected || staleFrames.length === 0) return null;
        return stale.get(frameKey(selected.id, staleFrames[0])) ?? null;
    }, [selected, staleFrames, stale]);

    /**
     * The frames that will seed a refinement run.
     *
     * Verified frames inside the range that still carry a mask: the anchor travels
     * separately, and a *cleared* frame has no mask to send (absence is not
     * conditionable), so both are left out.
     */
    const pinsFor = useCallback(
        (objectId: number, from: number, to: number) => {
            const tracklet = clip.tracklets.find(
                (item) => item.id === objectId,
            );
            const pins = new Map<number, RawRle>();
            if (!tracklet || tracklet.maskFrames.first < 0) return pins;
            const start = Math.max(from + 1, tracklet.maskFrames.first);
            const end = Math.min(to, tracklet.maskFrames.last);
            for (let frame = start; frame <= end; frame++) {
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

    const propRunRef = useRef(propRun);
    propRunRef.current = propRun;

    const discardPropagation = useCallback((restoreFrame = false) => {
        propAbortRef.current?.abort();
        propAbortRef.current = null;
        propJobRef.current = [];
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
            if (next === "propagate" && !selectedHasMaskHere) return;
            if (next !== "review") {
                setPlaying(false);
            }
            if (next === "propagate" && !propStatus) {
                void fetchPropagateStatus().then(setPropStatus);
            }
            discardPropagation(true);
            setTool(next);
            setPaintMode("add");
            // Drafts survive a tool switch (§6.1): the prompt is what is specific to
            // a tool, not the work already drawn.
            resetTransient();
        },
        [
            selected,
            selectedHasMaskHere,
            samAvailable,
            method,
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
     */
    const clearMaskOnFrame = useCallback(() => {
        if (selectedId === null || !selected) return;
        if (clip.rawMaskAt(selected, frameIndex) === null) return;
        if (selected.maskFrames.count <= 1) {
            setLocalNotice({
                kind: "info",
                text: `That is the only frame with a mask. Redraw it, or delete the ${vocab.unit}.`,
            });
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
        setLocalNotice({
            kind: "info",
            text: `Cleared frame ${frameIndex + 1}.`,
        });
    }, [
        selectedId,
        selected,
        clip,
        frameIndex,
        markCorrected,
        refresh,
        resetTransient,
        vocab.unit,
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
        if (toolRef.current === "propagate") discardPropagationRef.current();
    }, [selectedId, resetTransient]);

    const changeMethod = useCallback(
        (next: DrawMethod) => {
            if (next === method) return;
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
        [method, samAvailable, resetPrompt, sam],
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
            }
            setClip(next);
            setSelectedId(created[created.length - 1] ?? null);
            refresh();
            discardDraft();
            // Creating objects is an Add-mask act, so stay there and let the next
            // one be drawn straight away. A text prompt fired from Edit mask is
            // the only other way in, and that still returns to Review.
            if (tool !== "addMask") setTool("review");
            setLocalNotice({
                kind: "success",
                text: `Added ${created.length} ${
                    created.length === 1 ? vocab.unit : `${vocab.unit}s`
                }.`,
            });
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
            vocab.unit,
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
            setTool("review");
            setLocalNotice({
                kind: finalMask ? "success" : "info",
                text: finalMask
                    ? `Saved mask on frame ${frameIndex + 1}.`
                    : stillThere
                      ? `Removed the mask on frame ${frameIndex + 1}.`
                      : `Removed ${vocab.unit} #${selectedId}.`,
            });
            return;
        }

        // ----- Add mask
        if (!finalMask) return;
        if (!semantic) {
            const { clip: next, tracklet } = clip.addTracklet(
                frameIndex,
                finalMask,
            );
            const chosenLabel = clip.labelById(newLabelId);
            setClip(
                chosenLabel ? next.setLabel(tracklet.id, chosenLabel.id) : next,
            );
            setSelectedId(tracklet.id);
            discardDraft();
            // Stay in Add mask. The next object is usually drawn immediately, and
            // returning to Select forced a click on the rail for every one.
            // `discardDraft` already cleared the draft, the clicks and the
            // polygon, so the bar is back to a clean prompt.
            setLocalNotice({
                kind: "success",
                text: `Added ${vocab.unit} #${tracklet.id}.`,
            });
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
                setLocalNotice({
                    kind: "success",
                    text: `Added to class "${targetClass.label}".`,
                });
            } else {
                const { clip: next, tracklet } = clip.addClass(
                    frameIndex,
                    finalMask,
                    label,
                );
                setClip(next);
                setSelectedId(tracklet.id);
                setLocalNotice({
                    kind: "success",
                    text: `Created class "${label}".`,
                });
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
        vocab.unit,
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
            setClip(next);
            refresh();
            setLocalNotice({
                kind: "info",
                text: `Removed ${vocab.unit} #${id}.`,
            });
        },
        [clip, refresh, vocab.unit],
    );

    /** The object named by the open delete confirmation, if any. */
    const pendingDeleteTarget = useMemo(
        () =>
            pendingDelete === null
                ? null
                : (clip.tracklets.find((t) => t.id === pendingDelete) ?? null),
        [clip.tracklets, pendingDelete],
    );

    const propagateModel = "SAM 3 tracker";
    const propagateAvailable = propStatus?.available ?? false;
    const propagateNote = propStatus?.available
        ? null
        : `The tracker is unavailable (${propStatus?.error ?? "unknown reason"}).`;
    const propDirectionLabel =
        propDirection === "both"
            ? "both ways"
            : propDirection === "forward"
              ? "forward only"
              : "backward only";

    const propAnchor = propRun ? propRun.anchor : frameIndex;
    const anchorMask = useMemo(
        () =>
            tool === "propagate" && selected && !propRun
                ? clip.rawMaskAt(selected, frameIndex)
                : null,
        [tool, selected, propRun, clip, frameIndex],
    );

    const propRange = useMemo(() => {
        const back = Math.max(0, Math.min(propBack, propAnchor));
        const forward = Math.max(
            0,
            Math.min(propForward, clip.frameCount - 1 - propAnchor),
        );
        // No frame cap any more: the backend splits a long range into overlapping
        // windows, so a run may cover the whole clip.
        return {
            back,
            forward,
            first: propAnchor - back,
            last: propAnchor + forward,
        };
    }, [propBack, propForward, propAnchor, clip.frameCount]);

    /**
     * The range chosen on the bar.
     *
     * The anchor is clamped inside it: it holds the mask being propagated, so a
     * range that excluded it could not run. Whole-clip stays the default, which is
     * why the checkbox is set rather than cleared when both ends are touched.
     */
    const setPropSpan = useCallback(
        (first: number, last: number) => {
            const clampedFirst = Math.max(0, Math.min(first, propAnchor));
            const clampedLast = Math.min(
                clip.frameCount - 1,
                Math.max(last, propAnchor),
            );
            setPropWholeClip(
                clampedFirst === 0 && clampedLast === clip.frameCount - 1,
            );
            setPropBack(Math.max(0, propAnchor - clampedFirst));
            setPropForward(Math.max(0, clampedLast - propAnchor));
        },
        [propAnchor, clip.frameCount],
    );

    /** Frames the running job has produced, for the bar's progress band. */
    const producedFrames = useMemo(
        () => (propRun ? new Set(propRun.masks.keys()) : null),
        [propRun],
    );

    /**
     * The selected object's frame states, for the timeline strip.
     *
     * One state per frame, in the spec's §8.1 vocabulary. A stored draft wins over
     * everything (it is the uncommitted thing on that frame), then staleness, then
     * the committed mask — verified frames read as the stronger mark. Drafts being
     * *created* have no object yet, so they fill only the frames with no other
     * state: the point of showing them at all is that unsaved work is never
     * invisible.
     */
    const timelineStates = useMemo<TimelineFrameState[]>(() => {
        const count = clip.frameCount;
        const states: TimelineFrameState[] = new Array(count).fill("none");
        const applyCreateDrafts = () => {
            for (const key of drafts.keys()) {
                const separator = key.indexOf(":");
                if (Number(key.slice(0, separator)) !== NEW_OBJECT_ID) continue;
                const frame = Number(key.slice(separator + 1));
                if (frame >= 0 && frame < count && states[frame] === "none")
                    states[frame] = "draft";
            }
        };
        if (!selected) {
            applyCreateDrafts();
            return states;
        }
        for (let frame = 0; frame < count; frame++) {
            const key = frameKey(selected.id, frame);
            if (drafts.has(key)) states[frame] = "draft";
            else if (stale.has(key)) states[frame] = "stale";
            else if (clip.rawMaskAt(selected, frame) !== null)
                states[frame] = verifiedFrames.has(key)
                    ? "verified"
                    : "accepted";
        }
        if (propRun) {
            // The run being reviewed is uncommitted too, whether or not its object
            // is the one selected.
            for (const frame of propRun.masks.keys()) {
                if (frame < 0 || frame >= count) continue;
                if (states[frame] === "none" || states[frame] === "stale")
                    states[frame] = "preview";
            }
            // A finished run that produced nothing on a frame it covered did not
            // find the object there (occlusion or off-screen).
            if (!propRun.live) {
                for (
                    let frame = propRun.first;
                    frame <= propRun.last && frame < count;
                    frame++
                ) {
                    if (frame < 0 || frame === propRun.anchor) continue;
                    if (propRun.masks.has(frame)) continue;
                    if (states[frame] === "none") states[frame] = "lost";
                }
            }
        }
        applyCreateDrafts();
        return states;
    }, [clip, selected, drafts, stale, verifiedFrames, propRun]);

    /** Window boundaries of the reviewed run, for the timeline's drift ticks. */
    const timelineBoundaries = useMemo(() => propRun?.windows ?? [], [propRun]);

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
                    ? clip.rawMaskAt(tracklet, frameIndex) !== null
                    : false;
            }),
        [propTargets, clip, frameIndex],
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
        const override = propOverrideRef.current;
        propOverrideRef.current = null;
        // A refinement overrides the panel's range: it walks forward from the
        // corrected frame, because that is the only direction the correction has
        // anything to say about, and it replaces what it covers.
        const anchor = override?.anchor ?? frameIndex;
        const first = override?.first ?? (propWholeClip ? 0 : propRange.first);
        const last =
            override?.last ??
            (propWholeClip ? clip.frameCount - 1 : propRange.last);
        const direction = override ? "forward" : propDirection;
        const pins = override?.pins ?? new Map<number, RawRle>();
        const writePolicy =
            override?.writePolicy ??
            (propSkipExisting ? "skip-existing" : "replace-range");
        const targets = override ? [override.target] : propReady;
        if (
            !override &&
            !propWholeClip &&
            propRange.back + propRange.forward === 0
        )
            return;
        if (targets.length === 0) {
            setPropError(
                `Nothing to propagate: no selected ${vocab.unit} has a mask on frame ${anchor + 1}.`,
            );
            return;
        }
        const skipped = propTargets.filter((id) => !propReady.includes(id));

        const controller = new AbortController();
        propAbortRef.current = controller;
        setPropagating(true);
        setPropError(
            skipped.length === 0
                ? null
                : `Skipped ${skipped.length} ${skipped.length === 1 ? vocab.unit : vocab.units} with no mask on frame ${anchor + 1}.`,
        );
        setPropRun(null);
        setPropQueue([]);
        propWatchRef.current = null;
        const entries: PropQueueEntry[] = [];
        try {
            // Queue every object up front. The queue lives on the backend, so it
            // keeps working through the batch even if this panel is closed.
            for (const id of targets) {
                const tracklet = clip.tracklets.find((item) => item.id === id);
                const mask = tracklet ? clip.rawMaskAt(tracklet, anchor) : null;
                if (!tracklet || !mask) continue;
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
                    writePolicy,
                    pinned: pins.size,
                    windows: [],
                    masks: new Map(),
                    newest: null,
                });
            }
            if (entries.length === 0) return;
            propJobRef.current = entries.map((entry) => entry.jobId);
            setPropQueue([...entries]);
            const since = new Map<string, number>();
            let watching = 0;
            let live = entries.map((entry) => entry.jobId);
            while (live.length > 0) {
                // Poll the whole batch in one pass. Jobs that are still queued
                // answer immediately with their position, so this stays cheap
                // even with a long queue.
                const states = await Promise.all(
                    live.map((jobId) =>
                        getPropagationJob(jobId, {
                            since: since.get(jobId) ?? -1,
                            signal: controller.signal,
                        }),
                    ),
                );
                for (const state of states) {
                    const entry = entries.find(
                        (item) => item.jobId === state.jobId,
                    );
                    if (!entry) continue;
                    for (const item of state.masks) {
                        entry.masks.set(item.frameIndex, item);
                        // The last frame of a batch is the one the tracker just
                        // wrote, so that is the frame the canvas follows.
                        entry.newest = item.frameIndex;
                        since.set(
                            state.jobId,
                            Math.max(
                                since.get(state.jobId) ?? -1,
                                item.frameIndex,
                            ),
                        );
                    }
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
                // A row picked by hand wins over that choice, or the next poll
                // would snap the canvas back while it is being reviewed.
                const shown =
                    entries.find(
                        (entry) => entry.jobId === propWatchRef.current,
                    ) ?? entries[watching];
                setPropQueue([...entries]);
                setPropRun(entryToRun(shown, propStatus));
                // Follow the tracker as it works. Jumping to the frame it just
                // wrote is what makes the run look like a playing video showing
                // the newest result, rather than a still frame on the anchor.
                if (shown.newest !== null) setFrameIndex(shown.newest);
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
            }
        }
    }, [
        propagating,
        propReady,
        propTargets,
        propRange,
        propWholeClip,
        propDirection,
        propSkipExisting,
        frameIndex,
        frames,
        clip,
        propStatus,
        vocab.unit,
        vocab.units,
    ]);

    /** Pull a queued run's masks onto the canvas; they are already in memory. */
    const watchQueueJob = useCallback(
        (jobId: string) => {
            const entry = propQueue.find((item) => item.jobId === jobId);
            if (!entry) return;
            propWatchRef.current = jobId;
            setPropRun(entryToRun(entry, propStatus));
            const landing = [...entry.masks.keys()]
                .sort((a, b) => a - b)
                .find((frame) => frame !== entry.anchor);
            if (landing !== undefined) setFrameIndex(landing);
        },
        [propQueue, propStatus],
    );

    /**
     * Re-run the tracker from a frame a human has just corrected.
     *
     * This is the same job the panel submits; what makes it a refinement is what
     * it is *seeded* with. The corrected frame becomes the anchor, every other
     * frame a human verified in the range travels as a pin, and the result
     * replaces the range — because those frames hold precisely the stale masks
     * this run exists to correct.
     *
     * Nothing from the run being replaced is used as input. That is deliberate:
     * anchoring a window on the previous window's own guess is how a mistake
     * travels the length of a clip, and the reference implementation refuses to
     * do it for the same reason.
     */
    const repropagateFrom = useCallback(
        (from: number) => {
            if (propagating) return;
            if (selectedId === null) return;
            const last = clip.frameCount - 1;
            if (from >= last) {
                setLocalNotice({
                    kind: "info",
                    text: "Nothing after the last frame to re-propagate.",
                });
                return;
            }
            const pins = pinsFor(selectedId, from, last);
            propOverrideRef.current = {
                target: selectedId,
                anchor: from,
                first: from,
                last,
                pins,
                writePolicy: "replace-range",
            };
            setPropBatch([]);
            setFrameIndex(from);
            setTool("propagate");
            void runPropagation();
        },
        [propagating, selectedId, clip.frameCount, pinsFor, runPropagation],
    );

    /** Jump to the next frame a correction invalidated. */
    const nextStale = useCallback(() => {
        const next = staleFrames.find((frame) => frame > frameIndex);
        if (next !== undefined) setFrameIndex(next);
    }, [staleFrames, frameIndex]);

    /** Queue or unqueue a tracklet: Shift-click in the tracklet list. */
    const togglePropBatch = useCallback((id: number) => {
        setPropBatch((current) =>
            current.includes(id)
                ? current.filter((item) => item !== id)
                : [...current, id],
        );
    }, []);

    /**
     * Stop one queued run.
     *
     * Its frames stay in the queue: a stopped run is still worth reviewing, it
     * just will not produce any more.
     */
    const cancelQueuedJob = useCallback((jobId: string) => {
        propJobRef.current = propJobRef.current.filter((id) => id !== jobId);
        void cancelPropagation(jobId).catch(() => undefined);
    }, []);

    const cancelRunningPropagation = useCallback(() => {
        // Stop the whole queue, not just the run being watched: that is what Stop
        // means once several objects have been queued.
        for (const jobId of propJobRef.current) {
            void cancelPropagation(jobId).catch(() => undefined);
        }
        propJobRef.current = [];
        propAbortRef.current?.abort();
        propAbortRef.current = null;
        setPropagating(false);
        setPropError(null);
        setLocalNotice({
            kind: "info",
            text: "Propagation cancelled.",
        });
    }, []);

    const propPreview =
        tool === "propagate" ? (propRun?.masks.get(frameIndex) ?? null) : null;
    const propSummary = useMemo(() => {
        // Keyed to the run being reviewed, not to the selected object: in a batch
        // those are two different objects as soon as another row is watched.
        if (!propRun) return null;
        const tracklet = clip.tracklets.find(
            (item) => item.id === propRun.trackletId,
        );
        if (!tracklet) return null;
        const found = [...propRun.masks.values()].filter((m) => m.area > 0);
        const conflicts = found.filter(
            (m) => clip.rawMaskAt(tracklet, m.frameIndex) !== null,
        );
        // A refinement replaces the range, which is the only thing that makes
        // sense for it: the frames it covers hold the stale masks it was run to
        // correct. A first pass uses the panel's checkbox, which is why the
        // checkbox still decides it there.
        const replace =
            propRun.writePolicy === "replace-range" || !propSkipExisting;
        // A frame a human verified is never overwritten, and that includes the
        // frames they *cleared*: clearing is a statement that the object is not
        // there, and a run must not fill it back in.
        const accepted = found.filter((m) => {
            if (isVerified(propRun.trackletId, m.frameIndex)) return false;
            if (replace) return true;
            return clip.rawMaskAt(tracklet, m.frameIndex) === null;
        });
        return {
            total: propRun.masks.size,
            found: found.length,
            empty: propRun.masks.size - found.length,
            conflicts: conflicts.length,
            replace,
            accepted,
        };
    }, [propRun, clip, propSkipExisting, isVerified]);

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

    const acceptPropagation = useCallback(() => {
        if (!propRun || !propSummary) return;
        const tracklet = clip.tracklets.find(
            (item) => item.id === propRun.trackletId,
        );
        if (!tracklet) return;
        let next = clip;
        for (const item of propSummary.accepted) {
            next = next.replaceMask(
                propRun.trackletId,
                item.frameIndex,
                item.rle,
            );
        }
        // A stale frame the tracker did not find the object on loses its mask: the
        // run was conditioned on the correction and still saw nothing there, so the
        // old mask is the drift this run exists to remove.
        const dropped: number[] = [];
        if (propSummary.replace) {
            for (let frame = propRun.first; frame <= propRun.last; frame++) {
                if (next.rawMaskAt(tracklet, frame) === null) continue;
                if (!stale.has(frameKey(propRun.trackletId, frame))) continue;
                if (propSummary.accepted.some((i) => i.frameIndex === frame))
                    continue;
                dropped.push(frame);
            }
            for (const frame of dropped)
                next = next.removeMask(propRun.trackletId, frame);
        }
        setClip(next);
        refresh();
        const count = propSummary.accepted.length;
        const stored = new Set(
            propSummary.accepted.map((item) => item.frameIndex),
        );
        // Those frames are no longer stale: they now hold the corrected result.
        setStale((current) => {
            const remaining = new Map(current);
            for (const frame of stored)
                remaining.delete(frameKey(propRun.trackletId, frame));
            for (const frame of dropped)
                remaining.delete(frameKey(propRun.trackletId, frame));
            return remaining;
        });
        const jobId = propRun.jobId;
        // Accepting one object must not disturb the rest of the batch, so only
        // the frames just stored are dropped from that run's entry.
        const remaining = propQueue.filter((entry) => entry.jobId !== jobId);
        setPropQueue((current) =>
            current.map((entry) =>
                entry.jobId === jobId
                    ? {
                          ...entry,
                          masks: new Map(
                              [...entry.masks].filter(
                                  ([frame]) => !stored.has(frame),
                              ),
                          ),
                      }
                    : entry,
            ),
        );
        // Move on to the next run with something to review; the queue keeps
        // working through the rest in the background either way.
        const nextEntry =
            remaining.find(
                (entry) => entry.state === "done" && entry.masks.size > 0,
            ) ?? remaining.find((entry) => entry.masks.size > 0);
        setPropRun(nextEntry ? entryToRun(nextEntry, propStatus) : null);
        if (!nextEntry && !propQueue.some(isLive)) setTool("review");
        setLocalNotice({
            kind: count > 0 ? "success" : "info",
            text:
                count > 0
                    ? `Tracked ${count} frame${count === 1 ? "" : "s"} for ${vocab.unit} #${tracklet.id}.`
                    : "Nothing stored: the range already had masks.",
        });
    }, [
        propRun,
        propSummary,
        propQueue,
        clip,
        refresh,
        propStatus,
        vocab.unit,
    ]);

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
            selectedHasMaskHere,
            hasPropRun: propRun !== null,
            refineFrom,
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
            repropagateFrom,
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
                    aria-label="Open another project"
                    title="Open another project"
                    onClick={() => {
                        if (confirmDraftsDiscarded()) onReset();
                    }}
                />

                <Button
                    variant="primary"
                    icon="download"
                    disabled={exportBusyId !== null}
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
                    canPropagate={selectedHasMaskHere}
                    propagateModel={propagateModel}
                    onToolChange={changeTool}
                    onRefreshStatus={refreshSam}
                />

                <section className={styles.videoCol}>
                    {tool === "propagate" && selected && (
                        <div className={styles.promptBar} role="toolbar">
                            <span className={styles.promptTitle}>
                                Propagate · {selected.label} #{selected.id} ·
                                from frame {propAnchor + 1}
                            </span>
                            <span className={styles.promptStatus}>
                                Shift-click to queue
                            </span>

                            {staleFrames.length > 0 && (
                                <>
                                    <span className={styles.promptError}>
                                        {staleFrames.length} stale frame
                                        {staleFrames.length === 1 ? "" : "s"}
                                    </span>
                                    <button
                                        type="button"
                                        className="btn"
                                        disabled={propagating}
                                        onClick={nextStale}
                                        title="Jump to the next frame a correction invalidated"
                                    >
                                        Next stale
                                    </button>
                                    <button
                                        type="button"
                                        className="btn"
                                        disabled={propagating}
                                        onClick={() => {
                                            // Keep the corrections, drop the marks
                                            // that say they are unverified.
                                            setStale(new Map());
                                            setLocalNotice({
                                                kind: "info",
                                                text: "Cleared the stale marks.",
                                            });
                                        }}
                                        title="Drop the stale marks without re-propagating"
                                    >
                                        Ignore
                                    </button>
                                    <button
                                        type="button"
                                        className="btn btnPrimary"
                                        disabled={
                                            propagating ||
                                            !propagateAvailable ||
                                            (refineFrom ?? frameIndex) >=
                                                clip.frameCount - 1
                                        }
                                        onClick={() =>
                                            repropagateFrom(
                                                refineFrom ?? frameIndex,
                                            )
                                        }
                                        title="Ctrl+Enter — re-run the tracker from the corrected frame, seeded with every frame you verified and nothing else"
                                    >
                                        Re-propagate from{" "}
                                        {(refineFrom ?? frameIndex) + 1}
                                    </button>
                                </>
                            )}

                            {!propRun ? (
                                <>
                                    <label className={styles.rangeField}>
                                        Back
                                        <input
                                            type="number"
                                            className={`${styles.promptInput} ${styles.numberInput}`}
                                            min={0}
                                            max={propAnchor}
                                            value={propBack}
                                            disabled={propagating}
                                            onChange={(event) =>
                                                setPropBack(
                                                    Math.max(
                                                        0,
                                                        Math.floor(
                                                            Number(
                                                                event.target
                                                                    .value,
                                                            ) || 0,
                                                        ),
                                                    ),
                                                )
                                            }
                                            title="Frames before the anchor to track (backward)"
                                        />
                                    </label>
                                    <label className={styles.rangeField}>
                                        Forward
                                        <input
                                            type="number"
                                            className={`${styles.promptInput} ${styles.numberInput}`}
                                            min={0}
                                            max={
                                                clip.frameCount - 1 - propAnchor
                                            }
                                            value={propForward}
                                            disabled={propagating}
                                            onChange={(event) =>
                                                setPropForward(
                                                    Math.max(
                                                        0,
                                                        Math.floor(
                                                            Number(
                                                                event.target
                                                                    .value,
                                                            ) || 0,
                                                        ),
                                                    ),
                                                )
                                            }
                                            title="Frames after the anchor to track (forward)"
                                        />
                                    </label>
                                    <label className={styles.rangeField}>
                                        <span>Direction</span>
                                        <select
                                            className={styles.promptInput}
                                            value={propDirection}
                                            onChange={(event) =>
                                                setPropDirection(
                                                    event.target
                                                        .value as PropagationDirection,
                                                )
                                            }
                                        >
                                            <option value="both">
                                                Both ways
                                            </option>
                                            <option value="forward">
                                                After the anchor
                                            </option>
                                            <option value="backward">
                                                Before the anchor
                                            </option>
                                        </select>
                                    </label>
                                    <label className={styles.rangeField}>
                                        <span>Range</span>
                                        <input
                                            type="checkbox"
                                            checked={propWholeClip}
                                            onChange={(event) =>
                                                setPropWholeClip(
                                                    event.target.checked,
                                                )
                                            }
                                            title="Propagate over the whole clip instead of a fixed number of frames"
                                        />
                                        <span className={styles.promptStatus}>
                                            whole clip
                                        </span>
                                    </label>
                                    <span className={styles.promptStatus}>
                                        {propWholeClip
                                            ? `Frames 1–${clip.frameCount} (${propDirectionLabel})`
                                            : `Frames ${propRange.first + 1}–${propRange.last + 1} (${propDirectionLabel})`}
                                        {propStatus && propagateAvailable
                                            ? ` · ${propagateModel} on ${propStatus.device}`
                                            : ""}
                                    </span>
                                    {propagateNote && (
                                        <span className={styles.promptStatus}>
                                            {propagateNote}
                                        </span>
                                    )}
                                    <PropagationRange
                                        frameCount={clip.frameCount}
                                        anchor={propAnchor}
                                        first={propRange.first}
                                        last={propRange.last}
                                        direction={propDirection}
                                        produced={producedFrames}
                                        running={propagating}
                                        onChange={setPropSpan}
                                    />
                                    {!anchorMask && (
                                        <span className={styles.promptError}>
                                            {vocab.unit
                                                .charAt(0)
                                                .toUpperCase() +
                                                vocab.unit.slice(1)}{" "}
                                            #{selected.id} has no mask on this
                                            frame — move to a frame where it
                                            does.
                                        </span>
                                    )}
                                    {propStatus && !propagateAvailable && (
                                        <span className={styles.promptError}>
                                            {propagateModel} unavailable:{" "}
                                            {propStatus.error ??
                                                "unknown reason"}
                                        </span>
                                    )}
                                    {propError && (
                                        <span className={styles.promptError}>
                                            {propError}
                                        </span>
                                    )}
                                    {propagating && (
                                        <span className={styles.promptStatus}>
                                            Propagating
                                            {jobProgressText(propRun)}
                                            <button
                                                type="button"
                                                className="btn"
                                                onClick={
                                                    cancelRunningPropagation
                                                }
                                            >
                                                Cancel
                                            </button>
                                        </span>
                                    )}
                                    <span className={styles.spacer} />
                                    {propagating ? (
                                        <button
                                            type="button"
                                            className="btn"
                                            onClick={() => discardPropagation()}
                                            title="Stop the running propagation"
                                        >
                                            Stop
                                        </button>
                                    ) : (
                                        <button
                                            type="button"
                                            className="btn"
                                            onClick={() => changeTool("review")}
                                            title="Esc"
                                        >
                                            Cancel
                                        </button>
                                    )}
                                    <button
                                        type="button"
                                        className="btn btnPrimary"
                                        disabled={
                                            propagating ||
                                            propReady.length === 0 ||
                                            !propagateAvailable ||
                                            propRange.back +
                                                propRange.forward ===
                                                0
                                        }
                                        onClick={() => void runPropagation()}
                                        title={
                                            propReady.length > 1
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
                                </>
                            ) : (
                                <>
                                    <span className={styles.promptStatus}>
                                        {propRun.backend === "sam3"
                                            ? "SAM 3"
                                            : "SAM 2"}{" "}
                                        · {propSummary?.total ?? 0} frames in{" "}
                                        {(propRun.elapsedMs / 1000).toFixed(1)}{" "}
                                        s · {propSummary?.found ?? 0} with a
                                        mask
                                        {propSummary && propSummary.empty > 0
                                            ? `, ${propSummary.empty} empty (left unchanged)`
                                            : ""}
                                    </span>
                                    <span className={styles.promptStatus}>
                                        {frameIndex === propRun.anchor
                                            ? "Anchor frame (unchanged)"
                                            : propPreview
                                              ? propPreview.area > 0
                                                  ? `Frame ${frameIndex + 1}: ${propPreview.area.toLocaleString()} px${
                                                        clip.rawMaskAt(
                                                            selected,
                                                            frameIndex,
                                                        )
                                                            ? propSkipExisting
                                                                ? " · has a mask, will be skipped"
                                                                : " · replaces the existing mask"
                                                            : ""
                                                    }`
                                                  : `Frame ${frameIndex + 1}: nothing found`
                                              : `Frame ${frameIndex + 1} is outside the range ${propRun.first + 1}–${propRun.last + 1}`}
                                    </span>
                                    {propSummary &&
                                        propSummary.conflicts > 0 && (
                                            <label
                                                className={styles.rangeField}
                                            >
                                                <input
                                                    type="checkbox"
                                                    checked={propSkipExisting}
                                                    onChange={(event) =>
                                                        setPropSkipExisting(
                                                            event.target
                                                                .checked,
                                                        )
                                                    }
                                                />
                                                Skip {propSummary.conflicts}{" "}
                                                frame
                                                {propSummary.conflicts === 1
                                                    ? ""
                                                    : "s"}{" "}
                                                that already{" "}
                                                {propSummary.conflicts === 1
                                                    ? "has"
                                                    : "have"}{" "}
                                                a mask
                                            </label>
                                        )}
                                    <span className={styles.spacer} />
                                    <button
                                        type="button"
                                        className="btn"
                                        onClick={() => stepPropagated(-1)}
                                        title="Previous propagated frame (← also steps frames)"
                                    >
                                        ◀ Prev
                                    </button>
                                    <button
                                        type="button"
                                        className="btn"
                                        onClick={() => stepPropagated(1)}
                                        title="Next propagated frame (→ also steps frames)"
                                    >
                                        Next ▶
                                    </button>
                                    <button
                                        type="button"
                                        className="btn"
                                        onClick={() => discardPropagation(true)}
                                        title="Forget this result and set up another run"
                                    >
                                        Discard
                                    </button>
                                    <button
                                        type="button"
                                        className="btn"
                                        onClick={() => changeTool("review")}
                                        title="Esc"
                                    >
                                        Cancel
                                    </button>
                                    <button
                                        type="button"
                                        className="btn btnPrimary"
                                        disabled={
                                            !propSummary ||
                                            propSummary.accepted.length === 0
                                        }
                                        onClick={acceptPropagation}
                                        title="Enter"
                                    >
                                        Accept{" "}
                                        {propSummary?.accepted.length ?? 0} mask
                                        {propSummary?.accepted.length === 1
                                            ? ""
                                            : "s"}
                                    </button>
                                </>
                            )}
                        </div>
                    )}
                    {tool === "propagate" && (
                        <PropagationQueue
                            entries={propQueue}
                            watchingId={propRun?.jobId ?? null}
                            onWatch={watchQueueJob}
                            onCancel={cancelQueuedJob}
                            onCancelAll={cancelRunningPropagation}
                        />
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
                                {(
                                    [
                                        ["point", modelName],
                                        ["box", "Box"],
                                        ["text", "Text"],
                                        ["polygon", "Polygon"],
                                        ["brush", "Brush"],
                                    ] as [DrawMethod, string][]
                                ).map(([value, label]) => (
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
                                        {label}
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
                                        onClick={clearMaskOnFrame}
                                        title="Delete the mask on this frame and draw it again. A redraw then replaces it, instead of being added to the mask you are rejecting."
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

                            {semantic && tool === "addMask" && (
                                <>
                                    <input
                                        className={styles.promptInput}
                                        list="vsr-class-names"
                                        placeholder="class name (e.g. coral)"
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
                                        aria-label="Class name"
                                        title={
                                            method === "point" ||
                                            method === "box" ||
                                            method === "text"
                                                ? "Sent to SAM 3 as the concept to find. Matches an existing class by name, otherwise a new class is created. Enter runs the model."
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
                                            title="Run SAM 3 with the current clicks and class name (Enter in the name box)"
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
                                ) : method === "point" ||
                                  method === "box" ||
                                  method === "text" ? (
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
                        boundaries={timelineBoundaries}
                        label={
                            selected
                                ? `${vocab.unit} #${selected.id} · ${selected.label}`
                                : `No ${vocab.unit} selected`
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
                        draft={draftDecoded}
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
                        onDelete={setPendingDelete}
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
                    title={`Delete ${vocab.unit} #${pendingDelete}?`}
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
                                    deleteTracklet(pendingDelete);
                                    setPendingDelete(null);
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
