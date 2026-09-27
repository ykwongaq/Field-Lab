import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Clip } from "../lib/clip";
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
	fetchSamStatus,
	segmentConcept,
	segmentFrame,
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
import { LockIcon } from "./CreateProjectPanel";
import styles from "./Workspace.module.css";

export interface WorkspaceNotice {
	kind: "success" | "info";
	text: string;
}

interface WorkspaceProps {
	/** The clip as read from the archive; edits live in Workspace state. */
	clip: Clip;
	zip: ZipArchive;
	notice?: WorkspaceNotice | null;
	onDismissNotice?: () => void;
	onReset: () => void;
}

export function Workspace({
	clip: initialClip,
	zip,
	notice: externalNotice = null,
	onDismissNotice,
	onReset,
}: WorkspaceProps) {
	// The clip is immutable; mask edits replace it (see Clip.addTracklet etc.).
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

	// ----- Add / Edit mask tools -----
	// A mask is built as a *draft* (RLE) from any mix of operations: SAM
	// results, closed polygons and brush strokes, each added to or erased
	// from the draft. "Add mask" starts from an empty draft, "Edit mask" from
	// the selected tracklet's mask on the current frame. SAM 2 serves
	// instance projects, SAM 3 semantic ones.
	const semantic = clip.mode === "semantic";
	const modelName = semantic ? "SAM 3" : "SAM 2";
	const [tool, setTool] = useState<Tool>("review");
	const [method, setMethod] = useState<DrawMethod>("sam");
	const [paintMode, setPaintMode] = useState<PaintMode>("add");
	const [brushSize, setBrushSize] = useState(24);
	const [draft, setDraft] = useState<RawRle | null>(null);
	const [history, setHistory] = useState<(RawRle | null)[]>([]);
	const [polygon, setPolygon] = useState<FramePoint[]>([]);
	// SAM: clicks placed so far and the live result they produce. The result
	// is a *candidate* until it is applied to (or committed with) the draft.
	const [prompt, setPrompt] = useState<PromptPoint[]>([]);
	const [candidate, setCandidate] = useState<
		SamSegmentResult | Sam3SegmentResult | null
	>(null);
	const [segmenting, setSegmenting] = useState(false);
	const [promptError, setPromptError] = useState<string | null>(null);
	const segmentAbortRef = useRef<AbortController | null>(null);
	const [sam, setSam] = useState<SamStatus | null>(null);
	// Semantic only: class name typed in the prompt bar, sent to SAM 3 as the
	// concept and used to pick the class the mask is added to.
	const [className, setClassName] = useState("");
	const [exporting, setExporting] = useState(false);
	const [localNotice, setLocalNotice] = useState<WorkspaceNotice | null>(null);
	const notice = localNotice ?? externalNotice;
	const dismissNotice = useCallback(() => {
		if (localNotice) setLocalNotice(null);
		else onDismissNotice?.();
	}, [localNotice, onDismissNotice]);

	const fetchStatus = semantic ? fetchSam3Status : fetchSamStatus;
	const refreshSam = useCallback(() => {
		setSam(null);
		void fetchStatus().then(setSam);
	}, [fetchStatus]);
	useEffect(() => {
		const controller = new AbortController();
		void fetchStatus(controller.signal).then(setSam, () => {});
		return () => controller.abort();
	}, [fetchStatus]);

	// Warn before the tab closes with unexported mask edits.
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
		() => clip.tracklets.find((tracklet) => tracklet.id === selectedId) ?? null,
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
			const tracklet = clip.tracklets.find((candidate) => candidate.id === id);
			if (tracklet && tracklet.maskFrames.first >= 0) {
				setFrameIndex(tracklet.maskFrames.first);
			}
		},
		[clip.tracklets],
	);

	/** Clicking a mask on the frame selects it without moving the playhead. */
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
		const start = tracklets.findIndex((tracklet) => tracklet.id === selectedId);
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

	/** Drop the SAM clicks and their candidate (the draft is kept). */
	const resetPrompt = useCallback(() => {
		segmentAbortRef.current?.abort();
		segmentAbortRef.current = null;
		setPrompt([]);
		setCandidate(null);
		setSegmenting(false);
		setPromptError(null);
	}, []);

	/** Mask the tool started from: the selected tracklet's mask when editing. */
	const originalMask = useMemo(
		() =>
			tool === "editMask" && selected
				? clip.rawMaskAt(selected, frameIndex)
				: null,
		[tool, selected, clip, frameIndex],
	);

	/** Start over on the current frame: empty draft (add) or stored mask (edit). */
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

	// The class the semantic mask will go to: an existing class whose label
	// matches the typed name (case-insensitive), or a new one.
	const targetClass = useMemo(
		() => (semantic ? clip.findTrackletByLabel(className) : null),
		[semantic, clip, className],
	);
	const classLabels = useMemo(
		() => [...new Set(clip.tracklets.map((t) => t.label))].sort(),
		[clip.tracklets],
	);

	const samAvailable = sam?.available ?? false;

	const changeTool = useCallback(
		(next: Tool) => {
			if (next === "editMask" && !selected) return;
			if (next !== "review") {
				setPlaying(false);
				// Without the segmenter the tool still works with polygon/brush.
				if (!samAvailable && method === "sam") setMethod("polygon");
			}
			setTool(next);
			setPaintMode("add");
			resetDraft(next);
		},
		[selected, samAvailable, method, resetDraft],
	);

	// A draft belongs to one frame: moving the playhead starts over (for
	// Edit mask that means loading the selected tracklet's mask there).
	// `resetDraft` is read through a ref so only the frame change triggers this.
	const resetDraftRef = useRef(resetDraft);
	resetDraftRef.current = resetDraft;
	const toolRef = useRef(tool);
	toolRef.current = tool;
	useEffect(() => {
		if (toolRef.current !== "review") resetDraftRef.current();
	}, [frameIndex]);
	useEffect(() => {
		if (toolRef.current === "editMask") resetDraftRef.current();
	}, [selectedId]);

	/** Replace the draft, remembering the previous one for Undo. */
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
				const image = await zip.readAsBlob(clip.frameEntry(frameIndex));
				const request = {
					image,
					imageKey: `${clip.name}/${clip.frameNames[frameIndex]}`,
					points,
					signal: controller.signal,
				};
				// Semantic: SAM 3 segments the whole class (clicks + class name);
				// instance: SAM 2 segments the one clicked object.
				const result = semantic
					? await segmentConcept({ ...request, text })
					: await segmentFrame(request);
				if (controller.signal.aborted) return;
				setCandidate(result);
			} catch (cause) {
				if (controller.signal.aborted) return;
				setCandidate(null);
				setPromptError(cause instanceof Error ? cause.message : String(cause));
			} finally {
				if (segmentAbortRef.current === controller) {
					segmentAbortRef.current = null;
					setSegmenting(false);
				}
			}
		},
		[zip, clip, frameIndex, semantic, className],
	);

	// Semantic: run SAM 3 from the class name alone (Enter in the name box).
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

	/** SAM: fold the candidate into the draft and start a fresh prompt. */
	const applyCandidate = useCallback(() => {
		if (!candidate || segmenting) return;
		if (candidate.area > 0)
			pushDraft(composeRle(draft, candidate.rle, paintMode));
		resetPrompt();
	}, [candidate, segmenting, draft, paintMode, pushDraft, resetPrompt]);

	const addPolygonPoint = useCallback((point: FramePoint) => {
		setPolygon((current) => [...current, point]);
	}, []);

	/** Polygon: rasterise the closed shape into the draft. */
	const closePolygon = useCallback(() => {
		// A double click delivers two clicks first; drop a duplicated last vertex.
		const points = polygon.filter(
			(p, i) => i === 0 || p.x !== polygon[i - 1].x || p.y !== polygon[i - 1].y,
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

	/** Brush: a finished stroke, already rasterised by the video panel. */
	const applyStroke = useCallback(
		(stroke: RawRle, mode: PaintMode) => {
			setPromptError(null);
			pushDraft(composeRle(draft, stroke, mode));
		},
		[draft, pushDraft],
	);

	/** Backspace: last SAM click, last polygon vertex, or last draft change. */
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

	// What Commit would store: the draft plus a pending SAM candidate.
	const finalMask = useMemo(() => {
		const withCandidate =
			method === "sam" && candidate && candidate.area > 0
				? composeRle(draft, candidate.rle, paintMode)
				: draft;
		return withCandidate && rleArea(withCandidate) > 0 ? withCandidate : null;
	}, [method, candidate, draft, paintMode]);
	const draftDecoded = useMemo(
		() => (draft ? rleToDecoded(draft) : null),
		[draft],
	);
	const draftArea = useMemo(() => (draft ? rleArea(draft) : 0), [draft]);
	const draftChanged = draft !== originalMask || (candidate?.area ?? 0) > 0;

	const commitMask = useCallback(() => {
		if (tool === "review" || segmenting) return;

		// ----- Edit mask: replace the selected tracklet's mask on this frame.
		if (tool === "editMask") {
			if (selectedId === null || !selected || !draftChanged) return;
			const next = clip.replaceMask(selectedId, frameIndex, finalMask);
			const stillThere = next.tracklets.some((t) => t.id === selectedId);
			if (!stillThere) {
				store.remove(selectedId);
				const position = clip.tracklets.findIndex((t) => t.id === selectedId);
				const fallback =
					next.tracklets[Math.min(position, next.tracklets.length - 1)] ?? null;
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
			const { clip: next, tracklet } = clip.addTracklet(frameIndex, finalMask);
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

		// Semantic: the mask is the class's mask on this frame. Union it with
		// the class the name matches, or start a new class.
		const regions =
			method === "sam" && candidate && "instances" in candidate
				? `${candidate.instances} region${candidate.instances === 1 ? "" : "s"}`
				: `${rleArea(finalMask).toLocaleString()} px`;
		const label = className.trim() || NEW_TRACKLET_LABEL;
		try {
			if (targetClass) {
				const next = clip.paintClass(frameIndex, targetClass.id, finalMask);
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
			setPromptError(cause instanceof Error ? cause.message : String(cause));
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
				const position = clip.tracklets.findIndex((t) => t.id === selectedId);
				const fallback =
					next.tracklets[Math.min(position, next.tracklets.length - 1)] ?? null;
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

	const selectedHasMaskHere =
		selected !== null && clip.rawMaskAt(selected, frameIndex) !== null;

	// Frame-accurate playback loop driven by the clip's fps.
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
			if (target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName))
				return;

			if (tool !== "review") {
				switch (event.key) {
					case "Escape":
						event.preventDefault();
						changeTool("review");
						return;
					case "Enter":
						event.preventDefault();
						if (method === "polygon" && polygon.length >= 3) closePolygon();
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
						setBrushSize((size) => Math.max(1, Math.round(size / 1.25)));
						return;
					case "]":
						setBrushSize((size) => Math.min(200, Math.round(size * 1.25)));
						return;
				}
			}

			switch (event.key) {
				case "a":
					changeTool(tool === "addMask" ? "review" : "addMask");
					break;
				case "e":
					if (selected) changeTool(tool === "editMask" ? "review" : "editMask");
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
			// Semantic masks are per-frame label-map PNGs, so the whole project
			// archive is rewritten (frames/video copied, JSON + masks/ updated).
			setExporting(true);
			try {
				const blob = await clip.exportProjectZip(zip, store.getRecord());
				downloadBlob(`${clip.name}.zip`, blob);
				setLocalNotice({
					kind: "success",
					text: `Exported ${clip.name}.zip with ${clip.dirtyFrames.size} updated label map${clip.dirtyFrames.size === 1 ? "" : "s"}. Open it to continue from the saved state.`,
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
		// Instance: masks added/removed in the app: write the full annotation
		// JSON so it can replace `annotations/<clip>.json` inside the archive.
		downloadText(
			clip.annotationEntry.replace(/^annotations\//, ""),
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
								<span className={styles.modeLegacy}>assumed</span>
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
							{clip.tracklets.length === 1 ? vocab.unit : vocab.units}
						</span>
						{clip.videoEntry && (
							<span title={clip.videoEntry}>source video embedded</span>
						)}
						{missingFrames.length > 0 && (
							<span className={styles.warning}>
								{missingFrames.length} frame(s) missing from archive
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
					selectedMaskCount={selected?.maskFrames.count ?? 0}
					onToolChange={changeTool}
					onDelete={deleteSelected}
					onRefreshStatus={refreshSam}
				/>

				<section className={styles.videoCol}>
					{tool !== "review" && (
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
										disabled={value === "sam" && !samAvailable}
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
								{(["add", "erase"] as PaintMode[]).map((value) => (
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
								))}
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
											setBrushSize(Number(event.target.value))
										}
									/>
									<span className={styles.brushValue}>{brushSize}px</span>
								</label>
							)}

							{semantic && tool === "addMask" && (
								<>
									<input
										className={styles.promptInput}
										list="vsr-class-names"
										placeholder="class name (e.g. coral)"
										value={className}
										onChange={(event) => setClassName(event.target.value)}
										onKeyDown={(event) => {
											if (event.key === "Enter") {
												event.preventDefault();
												if (method === "sam") runTextPrompt();
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
												segmenting || (prompt.length === 0 && !className.trim())
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
									<span className={styles.promptError}>{promptError}</span>
								) : segmenting ? (
									"Segmenting…"
								) : candidate && method === "sam" ? (
									<>
										{"instances" in candidate && candidate.instances === 0 ? (
											<span className={styles.promptError}>
												No region above the SAM 3 threshold — add clicks, change
												the class name, or lower SAM3_THRESHOLD.
											</span>
										) : (
											<>
												{"instances" in candidate
													? `${candidate.instances} region${candidate.instances === 1 ? "" : "s"} · `
													: ""}
												score {candidate.score.toFixed(2)} ·{" "}
												{candidate.area.toLocaleString()} px
												{candidate.embeddingReused
													? ""
													: ` · encoder ${Math.round(candidate.encoderMs)} ms`}
												{draft ? " · Apply keeps it in the draft" : ""}
											</>
										)}
									</>
								) : draft ? (
									`Draft ${draftArea.toLocaleString()} px${
										tool === "editMask" && !draftChanged ? " (unchanged)" : ""
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
							{method === "sam" && candidate && candidate.area > 0 && (
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
									(tool === "editMask" ? !draftChanged : !finalMask)
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
						zip={zip}
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
						candidate={method === "sam" ? (candidate?.mask ?? null) : null}
						editingTrackletId={tool === "editMask" ? selectedId : null}
						onPromptPoint={addPromptPoint}
						onPolygonPoint={addPolygonPoint}
						onPolygonClose={closePolygon}
						onStroke={applyStroke}
						onSelectTracklet={selectOnCanvas}
						promptHint={
							semantic
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
						review={selectedId !== null ? store.get(selectedId) : null}
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
