import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Clip } from "../lib/clip";
import type { FrameSource } from "../lib/frames";
import { SessionFrameSource } from "../lib/frames";
import type { ZipArchive } from "../lib/zip";
import type {
    MaskVerdict,
    PromptPoint,
    RawRle,
    Taxonomy,
    TaxonomyKey,
} from "../types";
import { ReviewStore } from "../lib/review";
import { downloadText } from "../lib/format";
import { downloadBlob } from "../lib/zipWriter";
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
import { VideoPanel } from "./VideoPanel";
import { PropagationRange } from "./PropagationRange";
import {
    PropagationQueue,
    isLive,
    type PropQueueEntry,
} from "./PropagationQueue";
import { TrackletList } from "./TrackletList";
import { Inspector } from "./Inspector";
import {
    Toolbar,
    type DeleteScope,
    type DrawMethod,
    type Tool,
} from "./Toolbar";
import { LockIcon } from "./LockIcon";
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
        masks: entry.masks,
    };
}

/**
 * What a correction did to the frames after it.
 *
 * A correction is only consequential if something downstream becomes stale: on the
 * last frame of an object's run there is nothing to replace, and the edit simply
 * stands on its own.
 */
function refineHint(invalidated: number[], frame: number): string {
    if (invalidated.length === 0)
        return " Nothing after it needed re-checking, so it stays a single-frame edit.";
    return ` ${invalidated.length} frame${invalidated.length === 1 ? "" : "s"} after frame ${frame + 1} ${invalidated.length === 1 ? "is" : "are"} stale — re-propagate from here (Ctrl+Enter) to replace ${invalidated.length === 1 ? "it" : "them"}.`;
}

const DEFAULT_PROPAGATE_FORWARD = 10;

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
    const [clip, setClip] = useState(initialClip);
    const vocab = MODE_VOCABULARY[clip.mode];
    const [frameIndex, setFrameIndex] = useState(0);
    const [playing, setPlaying] = useState(false);
    const [selectedId, setSelectedId] = useState<number | null>(
        clip.tracklets[0]?.id ?? null,
    );
    const [showAllMasks, setShowAllMasks] = useState(false);
    const [maskOpacity, setMaskOpacity] = useState(0.55);
    const [tick, setTick] = useState(0);

    const semantic = clip.mode === "semantic";
    // There is one model now: SAM 3 answers text, box and point prompts, and the
    // same tracker propagates the mask it produced.
    const modelName = "SAM 3";
    const [tool, setTool] = useState<Tool>("review");
    const [method, setMethod] = useState<DrawMethod>("point");
    const [paintMode, setPaintMode] = useState<PaintMode>("add");
    const [brushSize, setBrushSize] = useState(24);
    const [draft, setDraft] = useState<RawRle | null>(null);
    const [history, setHistory] = useState<(RawRle | null)[]>([]);
    const [polygon, setPolygon] = useState<FramePoint[]>([]);

    const [prompt, setPrompt] = useState<PromptPoint[]>([]);
    const [box, setBox] = useState<PromptBox | null>(null);
    const [candidate, setCandidate] = useState<SegmentResult | null>(null);
    const [segmenting, setSegmenting] = useState(false);
    const [promptError, setPromptError] = useState<string | null>(null);
    const segmentAbortRef = useRef<AbortController | null>(null);
    const [sam, setSam] = useState<Sam3Status | null>(null);

    const [className, setClassName] = useState("");
    const [exporting, setExporting] = useState(false);
    const [localNotice, setLocalNotice] = useState<WorkspaceNotice | null>(
        null,
    );
    const notice = localNotice ?? externalNotice;

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
        if (clip.editCount === 0) return;
        const onBeforeUnload = (event: BeforeUnloadEvent) => {
            event.preventDefault();
        };
        window.addEventListener("beforeunload", onBeforeUnload);
        return () => window.removeEventListener("beforeunload", onBeforeUnload);
    }, [clip.editCount]);

    const storeRef = useRef<ReviewStore | null>(null);
    let store = storeRef.current;
    if (!store) {
        store = ReviewStore.load(clip.name);
        storeRef.current = store;
    }

    const refresh = useCallback(() => setTick((value) => value + 1), []);

    const selected = useMemo(
        () =>
            clip.tracklets.find((tracklet) => tracklet.id === selectedId) ??
            null,
        [clip.tracklets, selectedId],
    );

    const counts = useMemo(
        () => store.counts(clip.tracklets),
        [clip.tracklets, store, tick],
    );
    const missingFrames = useMemo(() => clip.missingFrames(zip), [clip, zip]);

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

    const setTaxonomyField = useCallback(
        (key: TaxonomyKey, value: string) => {
            if (selectedId === null || !selected) return;
            const current = store.get(selectedId);
            const next = { ...(current.taxonomy ?? selected.taxonomy) };
            next[key] = value;
            store.update(selectedId, {
                taxonomy: next,
                labelConfirmed: false,
            });
            refresh();
        },
        [selectedId, selected, store, refresh],
    );

    const applyTaxonomy = useCallback(
        (taxonomy: Taxonomy) => {
            if (selectedId === null) return;
            store.update(selectedId, {
                taxonomy,
                labelConfirmed: false,
            });
            refresh();
        },
        [selectedId, store, refresh],
    );

    const confirmLabel = useCallback(() => {
        if (selectedId === null) return;
        store.update(selectedId, { labelConfirmed: true });
        refresh();
    }, [selectedId, store, refresh]);

    const setMaskVerdict = useCallback(
        (verdict: MaskVerdict) => {
            if (selectedId === null) return;
            store.update(selectedId, { maskVerdict: verdict });
            refresh();
        },
        [selectedId, store, refresh],
    );

    const setComment = useCallback(
        (text: string) => {
            if (selectedId === null) return;
            store.update(selectedId, { comment: text });
            refresh();
        },
        [selectedId, store, refresh],
    );

    const nextUnverified = useCallback(() => {
        const tracklets = clip.tracklets;
        const total = tracklets.length;
        if (total === 0) return;
        const start = tracklets.findIndex(
            (tracklet) => tracklet.id === selectedId,
        );
        for (let k = 1; k <= total; k++) {
            const tracklet = tracklets[(start + k) % total];
            const review = store.get(tracklet.id);
            if (!review.labelConfirmed || !review.maskVerdict) {
                setSelectedId(tracklet.id);
                if (tracklet.maskFrames.first >= 0)
                    setFrameIndex(tracklet.maskFrames.first);
                return;
            }
        }
    }, [clip.tracklets, selectedId, store]);

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

    const originalMask = useMemo(
        () =>
            tool === "editMask" && selected
                ? clip.rawMaskAt(selected, frameIndex)
                : null,
        [tool, selected, clip, frameIndex],
    );

    const resetDraft = useCallback(
        (forTool: Tool = tool) => {
            resetPrompt();
            setPolygon([]);
            setHistory([]);
            setDraft(
                forTool === "editMask" && selected
                    ? clip.rawMaskAt(selected, frameIndex)
                    : null,
            );
        },
        [tool, selected, clip, frameIndex, resetPrompt],
    );

    const targetClass = useMemo(
        () => (semantic ? clip.findTrackletByLabel(className) : null),
        [semantic, clip, className],
    );
    const classLabels = useMemo(
        () => [...new Set(clip.tracklets.map((t) => t.label))].sort(),
        [clip.tracklets],
    );

    /** The key every per-frame state (draft, staleness, provenance) is held under. */
    const frameKey = (objectId: number, frame: number) =>
        `${objectId}:${frame}`;

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
            resetDraft(next);
        },
        [
            selected,
            selectedHasMaskHere,
            samAvailable,
            method,
            propStatus,
            discardPropagation,
            resetDraft,
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
                text: `Frame ${frameIndex + 1} is the only frame ${vocab.unit} #${selectedId} has a mask on, so there would be nothing left to correct. Redraw it instead, or delete the ${vocab.unit}.`,
            });
            return;
        }
        setClip(clip.removeMask(selectedId, frameIndex));
        const invalidated = markCorrected(selectedId, frameIndex);
        refresh();
        // Stay in Edit mask with an empty draft. The redraw then *replaces* the
        // mask on this frame; going through Add mask instead would create a second
        // object for the same thing, which is not a correction.
        resetDraft("review");
        setTool("editMask");
        setLocalNotice({
            kind: "info",
            text: `Cleared frame ${frameIndex + 1} of ${vocab.unit} #${selectedId}. Draw the new mask now (${modelName} click, box, text or polygon) and press Enter, or leave it cleared — that is recorded as "the object is not here".${refineHint(invalidated, frameIndex)}`,
        });
    }, [
        selectedId,
        selected,
        clip,
        frameIndex,
        markCorrected,
        refresh,
        resetDraft,
        vocab.unit,
    ]);

    const resetDraftRef = useRef(resetDraft);
    resetDraftRef.current = resetDraft;
    const toolRef = useRef(tool);
    toolRef.current = tool;
    useEffect(() => {
        if (toolRef.current !== "review" && toolRef.current !== "propagate")
            resetDraftRef.current();
    }, [frameIndex]);
    const discardPropagationRef = useRef(discardPropagation);
    discardPropagationRef.current = discardPropagation;
    useEffect(() => {
        if (toolRef.current === "editMask") resetDraftRef.current();
        if (toolRef.current === "propagate") discardPropagationRef.current();
    }, [selectedId]);

    const pushDraft = useCallback(
        (next: RawRle | null) => {
            setHistory((stack) => [...stack, draft].slice(-40));
            setDraft(next);
        },
        [draft],
    );

    const changeMethod = useCallback(
        (next: DrawMethod) => {
            if (next === method) return;
            const needsModel =
                next === "point" || next === "box" || next === "text";
            resetPrompt();
            setPolygon([]);
            setMethod(next);
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

    const applyCandidate = useCallback(() => {
        if (!candidate || segmenting) return;
        if (candidate.area > 0)
            pushDraft(composeRle(draft, candidate.rle, paintMode));
        resetPrompt();
    }, [candidate, segmenting, draft, paintMode, pushDraft, resetPrompt]);

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
        if (history.length === 0) return;
        setDraft(history[history.length - 1]);
        setHistory(history.slice(0, -1));
    }, [method, prompt.length, undoPromptPoint, polygon.length, history]);

    const canUndo =
        ((method === "point" || method === "box") && prompt.length > 0) ||
        (method === "polygon" && polygon.length > 0) ||
        history.length > 0;

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
        const withCandidate =
            (method === "point" || method === "box") &&
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
            let next = clip;
            const created: number[] = [];
            for (const instance of chosen) {
                const added = next.addTracklet(frameIndex, instance.rle, label);
                next = added.clip;
                created.push(added.tracklet.id);
            }
            setClip(next);
            setSelectedId(created[created.length - 1] ?? null);
            refresh();
            resetPrompt();
            resetDraft("review");
            setTool("review");
            setLocalNotice({
                kind: "success",
                text: `Added ${created.length} ${created.length === 1 ? vocab.unit : `${vocab.unit}s`} on frame ${frameIndex + 1}. Label ${
                    created.length === 1 ? "it" : "them"
                } in the inspector; use Export to save.`,
            });
        },
        [
            candidate,
            tool,
            segmenting,
            clip,
            frameIndex,
            className,
            refresh,
            resetPrompt,
            resetDraft,
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
                store.remove(selectedId);
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
            const invalidated = stillThere
                ? markCorrected(selectedId, frameIndex)
                : [];
            resetDraft("review");
            setTool("review");
            setLocalNotice({
                kind: finalMask ? "success" : "info",
                text: finalMask
                    ? `Corrected the mask of ${vocab.unit} "${selected.label}" (#${selectedId}) on frame ${frameIndex + 1}: ${rleArea(finalMask).toLocaleString()} px.${refineHint(invalidated, frameIndex)}`
                    : stillThere
                      ? `Removed the mask of ${vocab.unit} #${selectedId} on frame ${frameIndex + 1}.${refineHint(invalidated, frameIndex)}`
                      : `Removed ${vocab.unit} #${selectedId} (it had no other mask).`,
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
            setClip(next);
            setSelectedId(tracklet.id);
            resetDraft("review");
            setTool("review");
            setLocalNotice({
                kind: "success",
                text: `Added ${vocab.unit} #${tracklet.id} on frame ${frameIndex + 1}. Label it in the inspector; use Export to save the updated annotation JSON.`,
            });
            return;
        }

        const regionCount = candidate?.instances.length ?? 0;
        const regions =
            method === "text" && regionCount > 0
                ? `${regionCount} region${regionCount === 1 ? "" : "s"} merged`
                : `${rleArea(finalMask).toLocaleString()} px`;
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
                    text: `Added ${regions} to class "${targetClass.label}" on frame ${frameIndex + 1}. Use Export to save the updated project archive.`,
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
                    text: `Created class "${label}" with ${regions} on frame ${frameIndex + 1}. Use Export to save the updated project archive.`,
                });
            }
        } catch (cause) {
            setPromptError(
                cause instanceof Error ? cause.message : String(cause),
            );
            return;
        }
        resetDraft("review");
        setTool("review");
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
        resetDraft,
        markCorrected,
        vocab.unit,
        semantic,
        method,
        candidate,
        className,
        targetClass,
    ]);

    const deleteSelected = useCallback(
        (scope: DeleteScope) => {
            if (selectedId === null) return;
            const before = clip.tracklets.find((t) => t.id === selectedId);
            if (!before) return;
            const next =
                scope === "frame"
                    ? clip.removeMask(selectedId, frameIndex)
                    : clip.removeTracklet(selectedId);
            if (next === clip) return;
            const stillThere = next.tracklets.some((t) => t.id === selectedId);
            if (!stillThere) {
                store.remove(selectedId);
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
            setLocalNotice({
                kind: "info",
                text: stillThere
                    ? `Removed the mask of ${vocab.unit} #${selectedId} on frame ${frameIndex + 1}.`
                    : `Removed ${vocab.unit} #${selectedId}${
                          scope === "frame" ? " (it had no other mask)" : ""
                      }.`,
            });
        },
        [selectedId, clip, frameIndex, store, refresh, vocab.unit],
    );

    const propagateModel = "SAM 3 tracker";
    const propagateAvailable = propStatus?.available ?? false;
    const propagateNote = propStatus
        ? propStatus.available
            ? `Windows of ${propStatus.windowFrames} frames, ${propStatus.overlap} shared between them.`
            : `The tracker is unavailable (${propStatus.error ?? "unknown reason"}).`
        : null;
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
                    text: `Frame ${from + 1} is the last frame of the clip, so there is nothing after it to re-propagate.`,
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
            text: "Propagation cancelled. The frames produced so far are still here to review.",
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
        const skipped = propSummary.found - count;
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
        const { anchor, first, last } = propRun;
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
                    ? `Propagated ${vocab.unit} "${tracklet.label}" (#${tracklet.id}) from frame ${anchor + 1} to ${count} frame${count === 1 ? "" : "s"} (${first + 1}–${last + 1}) with ${propRun.backend === "sam3" ? "SAM 3" : "SAM 2"}${
                          skipped > 0
                              ? `; ${skipped} frame${skipped === 1 ? "" : "s"} kept ${skipped === 1 ? "its" : "their"} existing mask`
                              : ""
                      }${
                          propSummary.empty > 0
                              ? `; nothing found on ${propSummary.empty}`
                              : ""
                      }.${
                          nextEntry
                              ? " The next queued object is on the canvas."
                              : " Use Export to save."
                      }`
                    : "No mask was stored: every frame in the range already had one or the tracker found nothing.",
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

    useEffect(() => {
        const onKeyDown = (event: KeyboardEvent) => {
            const target = event.target as HTMLElement | null;
            if (
                target &&
                ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)
            )
                return;

            if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
                // Re-propagate from the corrected frame. Works from any tool, so a
                // reviewer who has just pressed Enter on a correction can press it
                // again without hunting for the propagate panel.
                if (refineFrom !== null) {
                    event.preventDefault();
                    repropagateFrom(refineFrom);
                }
                return;
            }

            if (tool === "propagate") {
                switch (event.key) {
                    case "Escape":
                        event.preventDefault();
                        changeTool("review");
                        return;
                    case "Enter":
                        event.preventDefault();
                        if (propRun) acceptPropagation();
                        else void runPropagation();
                        return;
                }
            } else if (tool !== "review") {
                switch (event.key) {
                    case "Escape":
                        event.preventDefault();
                        changeTool("review");
                        return;
                    case "Enter":
                        event.preventDefault();
                        if (method === "polygon" && polygon.length >= 3)
                            closePolygon();
                        else commitMask();
                        return;
                    case "Backspace":
                        event.preventDefault();
                        undo();
                        return;
                    case "s":
                        changeMethod("point");
                        return;
                    case "p":
                        changeMethod("polygon");
                        return;
                    case "b":
                        changeMethod("brush");
                        return;
                    case "[":
                        setBrushSize((size) =>
                            Math.max(1, Math.round(size / 1.25)),
                        );
                        return;
                    case "]":
                        setBrushSize((size) =>
                            Math.min(200, Math.round(size * 1.25)),
                        );
                        return;
                }
            }

            switch (event.key) {
                case "a":
                    changeTool(tool === "addMask" ? "review" : "addMask");
                    break;
                case "e":
                    if (selected)
                        changeTool(tool === "editMask" ? "review" : "editMask");
                    break;
                case "t":
                    if (selectedHasMaskHere || tool === "propagate")
                        changeTool(
                            tool === "propagate" ? "review" : "propagate",
                        );
                    break;
                case " ":
                    event.preventDefault();
                    setPlaying((value) => !value);
                    break;
                case "ArrowLeft":
                    event.preventDefault();
                    stepFrame(-1);
                    break;
                case "ArrowRight":
                    event.preventDefault();
                    stepFrame(1);
                    break;
                case "3":
                    setMaskVerdict("good");
                    break;
                case "4":
                    setMaskVerdict("bad");
                    break;
                case "n":
                    nextUnverified();
                    break;
                case "x":
                    setShowAllMasks((value) => !value);
                    break;
            }
        };
        window.addEventListener("keydown", onKeyDown);
        return () => window.removeEventListener("keydown", onKeyDown);
    }, [
        stepFrame,
        setMaskVerdict,
        nextUnverified,
        tool,
        method,
        polygon.length,
        selected,
        changeTool,
        changeMethod,
        closePolygon,
        commitMask,
        undo,
        selectedHasMaskHere,
        propRun,
        acceptPropagation,
        runPropagation,
        refineFrom,
        repropagateFrom,
    ]);

    const handleExport = useCallback(async () => {
        const payload = ReviewStore.buildExport(clip, store.getRecord());
        downloadText(
            `${clip.name}.review.json`,
            JSON.stringify(payload, null, 2),
            "application/json",
        );
        downloadText(`${clip.name}.review.csv`, payload.csv, "text/csv");
        if (clip.editCount === 0) return;
        if (semantic) {
            setExporting(true);
            try {
                const blob = await clip.exportProjectZip(
                    zip,
                    store.getRecord(),
                );
                downloadBlob(`${clip.name}.project`, blob);
                setLocalNotice({
                    kind: "success",
                    text: `Exported ${clip.name}.project with ${clip.dirtyFrames.size} updated label map${clip.dirtyFrames.size === 1 ? "" : "s"}. Open it to continue from the saved state.`,
                });
            } catch (cause) {
                setLocalNotice({
                    kind: "info",
                    text: `Export failed: ${cause instanceof Error ? cause.message : String(cause)}`,
                });
            } finally {
                setExporting(false);
            }
            return;
        }

        downloadText(
            `${clip.name}.json`,
            JSON.stringify(clip.toDataset(store.getRecord()), null, 2),
            "application/json",
        );
    }, [clip, store, semantic, zip]);

    return (
        <div className={styles.workspace}>
            <header className={styles.header}>
                <div>
                    <div className={styles.titleRow}>
                        <h1 className={styles.title}>{clip.name}</h1>
                        <span
                            className={`${styles.modeBadge} ${
                                clip.mode === "semantic"
                                    ? styles.modeSemantic
                                    : styles.modeInstance
                            }`}
                            title={`${vocab.title} segmentation — ${vocab.description} The mode was fixed when the project was created and cannot be changed.${
                                clip.modeAssumed
                                    ? " (Older archive without segmentation_mode in its JSON: assumed instance.)"
                                    : ""
                            }`}
                        >
                            <LockIcon size={11} />
                            {vocab.title}
                            {clip.modeAssumed && (
                                <span className={styles.modeLegacy}>
                                    assumed
                                </span>
                            )}
                        </span>
                    </div>
                    <div className={styles.meta}>
                        <span>{clip.frameCount} frames</span>
                        <span>{clip.fps} fps</span>
                        <span>
                            {clip.width}×{clip.height}
                        </span>
                        <span>
                            {clip.tracklets.length}{" "}
                            {clip.tracklets.length === 1
                                ? vocab.unit
                                : vocab.units}
                        </span>
                        {clip.videoEntry && (
                            <span title={clip.videoEntry}>
                                source video embedded
                            </span>
                        )}
                        {missingFrames.length > 0 && (
                            <span className={styles.warning}>
                                {missingFrames.length} frame(s) missing from
                                archive
                            </span>
                        )}
                        {clip.editCount > 0 && (
                            <span
                                className={styles.warning}
                                title="Mask edits made in this session are not in the archive yet. Export writes the updated annotation JSON."
                            >
                                {clip.editCount} unsaved mask edit
                                {clip.editCount === 1 ? "" : "s"}
                            </span>
                        )}
                    </div>
                </div>

                <div className={styles.spacer} />

                <div className={styles.progress}>
                    <span className={styles.progressText}>
                        {counts.verified} / {counts.total} verified
                    </span>
                    <div className={styles.progressTrack}>
                        <div
                            className={styles.progressFill}
                            style={{
                                width: counts.total
                                    ? `${Math.round((counts.verified / counts.total) * 100)}%`
                                    : "0%",
                            }}
                        />
                    </div>
                </div>

                <button
                    type="button"
                    className="btn btnPrimary"
                    onClick={() => void handleExport()}
                    disabled={exporting}
                >
                    {exporting ? "Exporting…" : "Export"}
                </button>
                <button type="button" className="btn" onClick={onReset}>
                    Open another
                </button>
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

            <div className={styles.body}>
                <Toolbar
                    mode={clip.mode}
                    tool={tool}
                    sam={sam}
                    modelName={modelName}
                    canDeleteFrame={selectedHasMaskHere}
                    canDeleteTracklet={selected !== null}
                    canEdit={selected !== null}
                    canPropagate={selectedHasMaskHere}
                    propagateModel={propagateModel}
                    selectedMaskCount={selected?.maskFrames.count ?? 0}
                    onToolChange={changeTool}
                    onDelete={deleteSelected}
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
                                Shift-click tracklets to queue several
                            </span>

                            {staleFrames.length > 0 && (
                                <>
                                    <span className={styles.promptError}>
                                        {staleFrames.length} frame
                                        {staleFrames.length === 1
                                            ? ""
                                            : "s"}{" "}
                                        {refineFrom !== null
                                            ? `after the correction on frame ${refineFrom + 1} are stale`
                                            : "are stale"}
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
                                                text: `Dropped the stale marks on ${vocab.unit} #${selected.id}. The masks themselves are untouched.`,
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

                            <div
                                className={styles.segmented}
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
                                        className={`${styles.segment} ${method === value ? styles.segmentActive : ""}`}
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

                            <div
                                className={styles.segmented}
                                role="radiogroup"
                                aria-label="Paint mode"
                            >
                                {(["add", "erase"] as PaintMode[]).map(
                                    (value) => (
                                        <button
                                            key={value}
                                            type="button"
                                            role="radio"
                                            aria-checked={paintMode === value}
                                            className={`${styles.segment} ${paintMode === value ? styles.segmentActive : ""}`}
                                            onClick={() => setPaintMode(value)}
                                            title={
                                                value === "add"
                                                    ? "New strokes, polygons and SAM results are added to the draft"
                                                    : "New strokes, polygons and SAM results are removed from the draft"
                                            }
                                        >
                                            {value === "add" ? "Add" : "Erase"}
                                        </button>
                                    ),
                                )}
                            </div>

                            {selected && selectedHasMaskHere && (
                                <button
                                    type="button"
                                    className="btn"
                                    onClick={clearMaskOnFrame}
                                    title="Delete the mask on this frame and draw it again. A redraw then replaces it, instead of being added to the mask you are rejecting."
                                >
                                    Clear frame mask
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
                                                Nothing was found — add clicks,
                                                draw a box, or reword the text
                                                prompt.
                                            </span>
                                        ) : (
                                            <>
                                                {isTextSplit
                                                    ? `${candidate.instances.length} objects found · `
                                                    : ""}
                                                score{" "}
                                                {candidate.score.toFixed(2)} ·{" "}
                                                {candidate.area.toLocaleString()}{" "}
                                                px
                                                {candidate.embeddingReused
                                                    ? ""
                                                    : ` · encoder ${Math.round(candidate.encoderMs)} ms`}
                                                {draft
                                                    ? " · Apply keeps it in the draft"
                                                    : ""}
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
                                    `Draft ${draftArea.toLocaleString()} px${
                                        tool === "editMask" && !draftChanged
                                            ? " (unchanged)"
                                            : ""
                                    }`
                                ) : tool === "editMask" ? (
                                    originalMask ? (
                                        "Everything erased — “Remove mask on this frame” drops it from this " +
                                        vocab.unit +
                                        "; Undo brings it back."
                                    ) : (
                                        "No mask on this frame yet — draw one to extend the " +
                                        vocab.unit +
                                        " here."
                                    )
                                ) : method === "point" ||
                                  method === "box" ||
                                  method === "text" ? (
                                    semantic ? (
                                        "Click a region of the class (Shift/right-click excludes), or type its name and press Enter."
                                    ) : (
                                        "Click the object in the frame; Shift/right-click marks a region to exclude."
                                    )
                                ) : method === "polygon" ? (
                                    "Click around the region; Enter, double-click or right-click closes the polygon."
                                ) : (
                                    "Paint with the mouse; Shift or right button erases."
                                )}
                            </span>
                            <div className={styles.spacer} />
                            {(method === "point" ||
                                method === "box" ||
                                method === "text") &&
                                candidate &&
                                candidate.area > 0 && (
                                    <button
                                        type="button"
                                        className="btn"
                                        disabled={segmenting}
                                        onClick={applyCandidate}
                                        title="Keep this result in the draft and start a new prompt (e.g. to add another region)"
                                    >
                                        Apply
                                    </button>
                                )}
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
                            <button
                                type="button"
                                className="btn"
                                disabled={!canUndo}
                                onClick={undo}
                                title="Backspace"
                            >
                                Undo
                            </button>
                            <button
                                type="button"
                                className="btn"
                                disabled={!canUndo && !draftChanged}
                                onClick={() => resetDraft()}
                                title="Discard the draft and start over on this frame"
                            >
                                Clear
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
                    <VideoPanel
                        clip={clip}
                        frames={frames}
                        frameIndex={frameIndex}
                        playing={playing}
                        selectedTrackletId={selectedId}
                        showAllMasks={showAllMasks}
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
                        onShowAllMasksChange={setShowAllMasks}
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
                                    ? "Reviewing the tracker's masks · ← → step frames · Enter accepts · Esc cancels"
                                    : "Set the range above and press Propagate · Enter runs · Esc cancels"
                                : semantic
                                  ? "Click a region of the class to find all of it · Shift-click or right-click to exclude · or type a class name above"
                                  : undefined
                        }
                    />
                </section>

                <aside className={styles.sidebar}>
                    <TrackletList
                        clip={clip}
                        selectedId={selectedId}
                        reviews={store.getRecord()}
                        onSelect={selectTracklet}
                        batch={propBatch}
                        onToggleBatch={
                            tool === "propagate" ? togglePropBatch : undefined
                        }
                    />
                    <Inspector
                        mode={clip.mode}
                        tracklet={selected}
                        review={
                            selectedId !== null ? store.get(selectedId) : null
                        }
                        onTaxonomyField={setTaxonomyField}
                        onApplyTaxonomy={applyTaxonomy}
                        onConfirmLabel={confirmLabel}
                        onMaskVerdict={setMaskVerdict}
                        onComment={setComment}
                    />
                </aside>
            </div>
        </div>
    );
}
