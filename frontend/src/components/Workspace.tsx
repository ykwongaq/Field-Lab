import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Clip } from "../lib/clip";
import type { FrameSource } from "../lib/frames";
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
    fetchPropagateStatus,
    fetchSam3Status,
    fetchSamStatus,
    propagateMask,
    segmentConcept,
    segmentFrame,
    type PropagateBackend,
    type PropagateStatus,
    type PropagatedFrame,
    type Sam3SegmentResult,
    type SamSegmentResult,
    type SamStatus,
} from "../lib/samApi";
import { VideoPanel } from "./VideoPanel";
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
    trackletId: number;
    anchor: number;
    first: number;
    last: number;
    backend: PropagateBackend;
    model: string;
    device: string;
    elapsedMs: number;
    masks: Map<number, PropagatedFrame>;
}

const DEFAULT_PROPAGATE_FORWARD = 10;

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
    const modelName = semantic ? "SAM 3" : "SAM 2";
    const [tool, setTool] = useState<Tool>("review");
    const [method, setMethod] = useState<DrawMethod>("sam");
    const [paintMode, setPaintMode] = useState<PaintMode>("add");
    const [brushSize, setBrushSize] = useState(24);
    const [draft, setDraft] = useState<RawRle | null>(null);
    const [history, setHistory] = useState<(RawRle | null)[]>([]);
    const [polygon, setPolygon] = useState<FramePoint[]>([]);

    const [prompt, setPrompt] = useState<PromptPoint[]>([]);
    const [candidate, setCandidate] = useState<
        SamSegmentResult | Sam3SegmentResult | null
    >(null);
    const [segmenting, setSegmenting] = useState(false);
    const [promptError, setPromptError] = useState<string | null>(null);
    const segmentAbortRef = useRef<AbortController | null>(null);
    const [sam, setSam] = useState<SamStatus | null>(null);

    const [className, setClassName] = useState("");
    const [exporting, setExporting] = useState(false);
    const [localNotice, setLocalNotice] = useState<WorkspaceNotice | null>(
        null,
    );
    const notice = localNotice ?? externalNotice;

    const [propStatus, setPropStatus] = useState<PropagateStatus | null>(null);
    const [propBack, setPropBack] = useState(0);
    const [propForward, setPropForward] = useState(DEFAULT_PROPAGATE_FORWARD);
    const [propagating, setPropagating] = useState(false);
    const [propError, setPropError] = useState<string | null>(null);
    const [propRun, setPropRun] = useState<PropagateRun | null>(null);
    const [propSkipExisting, setPropSkipExisting] = useState(true);
    const propAbortRef = useRef<AbortController | null>(null);
    const dismissNotice = useCallback(() => {
        if (localNotice) setLocalNotice(null);
        else onDismissNotice?.();
    }, [localNotice, onDismissNotice]);

    const fetchStatus = semantic ? fetchSam3Status : fetchSamStatus;
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

    const samAvailable = sam?.available ?? false;

    const selectedHasMaskHere =
        selected !== null && clip.rawMaskAt(selected, frameIndex) !== null;

    const propRunRef = useRef(propRun);
    propRunRef.current = propRun;

    const discardPropagation = useCallback((restoreFrame = false) => {
        propAbortRef.current?.abort();
        propAbortRef.current = null;
        if (restoreFrame && propRunRef.current)
            setFrameIndex(propRunRef.current.anchor);
        setPropagating(false);
        setPropRun(null);
        setPropError(null);
    }, []);

    const changeTool = useCallback(
        (next: Tool) => {
            if (next === "editMask" && !selected) return;
            if (next === "propagate" && !selectedHasMaskHere) return;
            if (next !== "review") {
                setPlaying(false);
                if (!samAvailable && method === "sam") setMethod("polygon");
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
            if (next === "sam" && !samAvailable) return;
            resetPrompt();
            setPolygon([]);
            setMethod(next);
        },
        [method, samAvailable, resetPrompt],
    );

    const runSegmentation = useCallback(
        async (points: PromptPoint[], text: string = className) => {
            segmentAbortRef.current?.abort();
            const controller = new AbortController();
            segmentAbortRef.current = controller;
            setSegmenting(true);
            setPromptError(null);
            try {
                const image = await frames.frame(frameIndex);
                const request = {
                    image,
                    imageKey: `${clip.name}/${clip.frameNames[frameIndex]}`,
                    points,
                    signal: controller.signal,
                };

                const result = semantic
                    ? await segmentConcept({ ...request, text })
                    : await segmentFrame(request);
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
        [frames, clip, frameIndex, semantic, className],
    );

    const runTextPrompt = useCallback(() => {
        if (!semantic || method !== "sam") return;
        if (prompt.length === 0 && !className.trim()) return;
        void runSegmentation(prompt, className);
    }, [semantic, method, prompt, className, runSegmentation]);

    const addPromptPoint = useCallback(
        (point: PromptPoint) => {
            const next = [...prompt, point];
            setPrompt(next);
            void runSegmentation(next);
        },
        [prompt, runSegmentation],
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
            void runSegmentation(next);
        }
    }, [prompt, runSegmentation]);

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
        if (method === "sam" && prompt.length > 0) {
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
        (method === "sam" && prompt.length > 0) ||
        (method === "polygon" && polygon.length > 0) ||
        history.length > 0;

    const finalMask = useMemo(() => {
        const withCandidate =
            method === "sam" && candidate && candidate.area > 0
                ? composeRle(draft, candidate.rle, paintMode)
                : draft;
        return withCandidate && rleArea(withCandidate) > 0
            ? withCandidate
            : null;
    }, [method, candidate, draft, paintMode]);
    const draftDecoded = useMemo(
        () => (draft ? rleToDecoded(draft) : null),
        [draft],
    );
    const draftArea = useMemo(() => (draft ? rleArea(draft) : 0), [draft]);
    const draftChanged = draft !== originalMask || (candidate?.area ?? 0) > 0;

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
            resetDraft("review");
            setTool("review");
            setLocalNotice({
                kind: finalMask ? "success" : "info",
                text: finalMask
                    ? `Updated the mask of ${vocab.unit} "${selected.label}" (#${selectedId}) on frame ${frameIndex + 1}: ${rleArea(finalMask).toLocaleString()} px. Use Export to save.`
                    : stillThere
                      ? `Removed the mask of ${vocab.unit} #${selectedId} on frame ${frameIndex + 1}.`
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

        const regions =
            method === "sam" && candidate && "instances" in candidate
                ? `${candidate.instances} region${candidate.instances === 1 ? "" : "s"}`
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

    const propagateBackend: PropagateBackend =
        semantic && propStatus?.sam3.available ? "sam3" : "sam2";
    const propagateModel = propagateBackend === "sam3" ? "SAM 3" : "SAM 2";
    const propagateAvailable = propStatus
        ? propStatus[propagateBackend].available
        : false;
    const propagateFallbackNote =
        semantic && propStatus && !propStatus.sam3.available
            ? `SAM 3 tracker unavailable (${propStatus.sam3.error ?? "unknown reason"}); using the SAM 2 video predictor instead.`
            : null;

    const propAnchor = propRun ? propRun.anchor : frameIndex;
    const anchorMask = useMemo(
        () =>
            tool === "propagate" && selected && !propRun
                ? clip.rawMaskAt(selected, frameIndex)
                : null,
        [tool, selected, propRun, clip, frameIndex],
    );

    const propRange = useMemo(() => {
        let back = Math.max(0, Math.min(propBack, propAnchor));
        let forward = Math.max(
            0,
            Math.min(propForward, clip.frameCount - 1 - propAnchor),
        );
        const cap = propStatus?.maxFrames ?? 0;
        if (cap > 0 && back + forward + 1 > cap) {
            forward = Math.max(0, Math.min(forward, cap - 1 - back));
            back = Math.max(0, Math.min(back, cap - 1 - forward));
        }
        return {
            back,
            forward,
            first: propAnchor - back,
            last: propAnchor + forward,
        };
    }, [propBack, propForward, propAnchor, clip.frameCount, propStatus]);
    const propCapped =
        propRange.back < Math.min(propBack, propAnchor) ||
        propRange.forward <
            Math.min(propForward, clip.frameCount - 1 - propAnchor);

    const runPropagation = useCallback(async () => {
        if (!selected || !anchorMask || propagating) return;
        if (propRange.back === 0 && propRange.forward === 0) return;
        const controller = new AbortController();
        propAbortRef.current = controller;
        setPropagating(true);
        setPropError(null);
        setPropRun(null);
        const anchor = frameIndex;
        try {
            const indices: number[] = [];
            for (
                let index = propRange.first;
                index <= propRange.last;
                index++
            ) {
                indices.push(index);
            }
            // Fetched together: these are round trips to the session, not local
            // slices of an archive.
            const images = await Promise.all(
                indices.map((index) => frames.frame(index)),
            );
            const result = await propagateMask({
                frames: indices.map((index, i) => ({
                    index,
                    image: images[i],
                })),
                anchor,
                mask: anchorMask,
                backward: propRange.back,
                forward: propRange.forward,
                backend: propagateBackend,
                signal: controller.signal,
            });
            if (controller.signal.aborted) return;
            setPropRun({
                trackletId: selected.id,
                anchor,
                first: propRange.first,
                last: propRange.last,
                backend: result.backend,
                model: result.model,
                device: result.device,
                elapsedMs: result.elapsedMs,
                masks: new Map(
                    result.masks.map((item) => [item.frameIndex, item]),
                ),
            });
            setFrameIndex(propRange.forward > 0 ? anchor + 1 : anchor - 1);
        } catch (cause) {
            if (controller.signal.aborted) return;
            setPropError(
                cause instanceof Error ? cause.message : String(cause),
            );
        } finally {
            if (propAbortRef.current === controller) {
                propAbortRef.current = null;
                setPropagating(false);
            }
        }
    }, [
        selected,
        anchorMask,
        propagating,
        propRange,
        frameIndex,
        frames,
        clip,
        propagateBackend,
    ]);

    const propPreview =
        tool === "propagate" ? (propRun?.masks.get(frameIndex) ?? null) : null;
    const propSummary = useMemo(() => {
        if (!propRun || !selected) return null;
        const found = [...propRun.masks.values()].filter((m) => m.area > 0);
        const conflicts = found.filter(
            (m) => clip.rawMaskAt(selected, m.frameIndex) !== null,
        );
        const accepted = propSkipExisting
            ? found.filter(
                  (m) => clip.rawMaskAt(selected, m.frameIndex) === null,
              )
            : found;
        return {
            total: propRun.masks.size,
            found: found.length,
            empty: propRun.masks.size - found.length,
            conflicts: conflicts.length,
            accepted,
        };
    }, [propRun, selected, clip, propSkipExisting]);

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
        if (!propRun || !propSummary || !selected || propagating) return;
        let next = clip;
        for (const item of propSummary.accepted) {
            next = next.replaceMask(
                propRun.trackletId,
                item.frameIndex,
                item.rle,
            );
        }
        setClip(next);
        refresh();
        const count = propSummary.accepted.length;
        const skipped = propSummary.found - count;
        discardPropagation();
        setTool("review");
        setLocalNotice({
            kind: count > 0 ? "success" : "info",
            text:
                count > 0
                    ? `Propagated ${vocab.unit} "${selected.label}" (#${selected.id}) from frame ${propRun.anchor + 1} to ${count} frame${count === 1 ? "" : "s"} (${propRun.first + 1}–${propRun.last + 1}) with ${propRun.backend === "sam3" ? "SAM 3" : "SAM 2"}${
                          skipped > 0
                              ? `; ${skipped} frame${skipped === 1 ? "" : "s"} kept ${skipped === 1 ? "its" : "their"} existing mask`
                              : ""
                      }${
                          propSummary.empty > 0
                              ? `; nothing found on ${propSummary.empty}`
                              : ""
                      }. Use Export to save.`
                    : "No mask was stored: every frame in the range already had one or the tracker found nothing.",
        });
    }, [
        propRun,
        propSummary,
        selected,
        propagating,
        clip,
        refresh,
        discardPropagation,
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
                        changeMethod("sam");
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
                                    <span className={styles.promptStatus}>
                                        Frames {propRange.first + 1}–
                                        {propRange.last + 1} (
                                        {propRange.back + propRange.forward} to
                                        fill) ·{" "}
                                        {propagateModel === "SAM 3"
                                            ? "SAM 3 tracker"
                                            : "SAM 2 video predictor"}
                                        {propStatus && propagateAvailable
                                            ? ` on ${propStatus[propagateBackend].device}`
                                            : ""}
                                        {propCapped && propStatus
                                            ? ` · capped at ${propStatus.maxFrames} frames per run`
                                            : ""}
                                    </span>
                                    {propagateFallbackNote && (
                                        <span className={styles.promptStatus}>
                                            {propagateFallbackNote}
                                        </span>
                                    )}
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
                                            {propagateModel} tracker
                                            unavailable:{" "}
                                            {propStatus[propagateBackend]
                                                .error ?? "unknown reason"}
                                        </span>
                                    )}
                                    {propError && (
                                        <span className={styles.promptError}>
                                            {propError}
                                        </span>
                                    )}
                                    {propagating && (
                                        <span className={styles.promptStatus}>
                                            Propagating{" "}
                                            {propRange.back +
                                                propRange.forward +
                                                1}{" "}
                                            frames…
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
                                            !anchorMask ||
                                            !propagateAvailable ||
                                            propRange.back +
                                                propRange.forward ===
                                                0
                                        }
                                        onClick={() => void runPropagation()}
                                        title="Enter"
                                    >
                                        {propagating
                                            ? "Propagating…"
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
                                        ["sam", modelName],
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
                                        disabled={
                                            value === "sam" && !samAvailable
                                        }
                                        onClick={() => changeMethod(value)}
                                        title={
                                            value === "sam"
                                                ? samAvailable
                                                    ? `Click prompts segmented by ${modelName} (S)`
                                                    : `${modelName} is unavailable${sam?.error ? `: ${sam.error}` : ""}`
                                                : value === "polygon"
                                                  ? "Click vertices, close with Enter, double-click or right-click (P)"
                                                  : "Drag to paint; Shift/right-drag erases; [ ] resize (B)"
                                        }
                                    >
                                        {label}
                                    </button>
                                ))}
                            </div>

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
                                                if (method === "sam")
                                                    runTextPrompt();
                                                else commitMask();
                                            } else if (event.key === "Escape") {
                                                event.preventDefault();
                                                changeTool("review");
                                            }
                                        }}
                                        aria-label="Class name"
                                        title={
                                            method === "sam"
                                                ? "Sent to SAM 3 as the concept to find. Matches an existing class by name, otherwise a new class is created. Enter runs the model."
                                                : "Class the drawn mask is added to (matched by name; otherwise a new class is created)."
                                        }
                                    />
                                    <datalist id="vsr-class-names">
                                        {classLabels.map((label) => (
                                            <option key={label} value={label} />
                                        ))}
                                    </datalist>
                                    {method === "sam" && (
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
                                ) : candidate && method === "sam" ? (
                                    <>
                                        {"instances" in candidate &&
                                        candidate.instances === 0 ? (
                                            <span
                                                className={styles.promptError}
                                            >
                                                No region above the SAM 3
                                                threshold — add clicks, change
                                                the class name, or lower
                                                SAM3_THRESHOLD.
                                            </span>
                                        ) : (
                                            <>
                                                {"instances" in candidate
                                                    ? `${candidate.instances} region${candidate.instances === 1 ? "" : "s"} · `
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
                                ) : method === "sam" ? (
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
                            {method === "sam" &&
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
                        onStep={stepFrame}
                        onShowAllMasksChange={setShowAllMasks}
                        onMaskOpacityChange={setMaskOpacity}
                        tool={tool}
                        method={method}
                        paintMode={paintMode}
                        brushSize={brushSize}
                        prompt={prompt}
                        polygon={polygon}
                        draft={draftDecoded}
                        candidate={
                            tool === "propagate"
                                ? (propPreview?.mask ?? null)
                                : method === "sam"
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
