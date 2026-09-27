import type { DecodedMask, PromptPoint, RawRle } from "../types";
import { API_BASE } from "./apiBase";

/**
 * Client for the SAM 2 endpoints.
 */

export interface SamStatus {
	available: boolean;
	loaded: boolean;
	model: string;
	device: string;
	error: string | null;
}

export interface SamSegmentResult {
	rle: RawRle;
	mask: DecodedMask;
	score: number;
	area: number;
	bbox: [number, number, number, number] | null;
	embeddingReused: boolean;
	encoderMs: number;
	decoderMs: number;
}

/** SAM 3 result: the class mask plus how many regions were merged into it. */
export interface Sam3SegmentResult extends SamSegmentResult {
	instances: number;
	instanceScores: number[];
	/** Exemplar boxes the clicks were turned into: [x0, y0, x1, y1, label]. */
	exemplars: number[][];
}

export class SamApiError extends Error {
	readonly status: number;

	constructor(status: number, message: string) {
		super(message);
		this.name = "SamApiError";
		this.status = status;
	}
}

const STATUS_ENDPOINT = `${API_BASE}/api/sam/status`;
const SEGMENT_ENDPOINT = `${API_BASE}/api/sam/segment`;
const SAM3_STATUS_ENDPOINT = `${API_BASE}/api/sam3/status`;
const SAM3_SEGMENT_ENDPOINT = `${API_BASE}/api/sam3/segment`;
const PROPAGATE_STATUS_ENDPOINT = `${API_BASE}/api/propagate/status`;
const PROPAGATE_ENDPOINT = `${API_BASE}/api/propagate`;

/** SAM 2 availability (instance projects). */
export function fetchSamStatus(signal?: AbortSignal): Promise<SamStatus> {
	return fetchStatus(STATUS_ENDPOINT, "SAM 2", signal);
}

/** SAM 3 availability (semantic projects). */
export function fetchSam3Status(signal?: AbortSignal): Promise<SamStatus> {
	return fetchStatus(SAM3_STATUS_ENDPOINT, "SAM 3", signal);
}

async function fetchStatus(
	endpoint: string,
	fallbackModel: string,
	signal?: AbortSignal,
): Promise<SamStatus> {
	try {
		const response = await fetch(endpoint, { signal });
		if (!response.ok) {
			return unavailable(
				fallbackModel,
				`Backend answered ${response.status} for ${fallbackModel} status.`,
			);
		}
		const payload = (await response.json()) as Partial<SamStatus>;
		return {
			available: Boolean(payload.available),
			loaded: Boolean(payload.loaded),
			model: payload.model ?? fallbackModel,
			device: payload.device ?? "?",
			error: payload.error ?? null,
		};
	} catch (cause) {
		if (cause instanceof DOMException && cause.name === "AbortError") throw cause;
		return unavailable(
			fallbackModel,
			"Backend is not reachable; start it with `uvicorn main:app`.",
		);
	}
}

function unavailable(model: string, error: string): SamStatus {
	return { available: false, loaded: false, model, device: "?", error };
}

export interface SegmentFrameOptions {
	/** The frame image exactly as stored in the archive. */
	image: Blob;
	/** Stable id of the frame so refinement clicks reuse the cached embedding. */
	imageKey: string;
	points: PromptPoint[];
	signal?: AbortSignal;
}

export async function segmentFrame(
	options: SegmentFrameOptions,
): Promise<SamSegmentResult> {
	if (options.points.length === 0) {
		throw new SamApiError(0, "At least one prompt point is required.");
	}
	const form = new FormData();
	form.append("image", options.image, "frame.jpg");
	form.append("points", JSON.stringify(options.points));
	form.append("image_key", options.imageKey);

	const response = await fetch(SEGMENT_ENDPOINT, {
		method: "POST",
		body: form,
		signal: options.signal,
	});
	if (!response.ok) {
		throw new SamApiError(response.status, await readDetail(response));
	}
	const payload = (await response.json()) as {
		rle: RawRle;
		runs: DecodedMask["runs"];
		height: number;
		width: number;
		score: number;
		area: number;
		bbox: [number, number, number, number] | null;
		embedding_reused: boolean;
		encoder_ms: number;
		decoder_ms: number;
	};
	return {
		rle: payload.rle,
		mask: { height: payload.height, width: payload.width, runs: payload.runs },
		score: payload.score,
		area: payload.area,
		bbox: payload.bbox,
		embeddingReused: payload.embedding_reused,
		encoderMs: payload.encoder_ms,
		decoderMs: payload.decoder_ms,
	};
}

export interface SegmentConceptOptions extends SegmentFrameOptions {
	/** Class name / noun phrase for SAM 3, e.g. "coral". Optional with clicks. */
	text?: string;
}

/**
 * SAM 3 concept segmentation of one frame. Needs at least one click or a
 * non-empty `text`; returns the union of all detected regions as one mask.
 */
export async function segmentConcept(
	options: SegmentConceptOptions,
): Promise<Sam3SegmentResult> {
	const text = options.text?.trim() ?? "";
	if (options.points.length === 0 && !text) {
		throw new SamApiError(0, "Click a region or type a class name first.");
	}
	const form = new FormData();
	form.append("image", options.image, "frame.jpg");
	form.append("points", JSON.stringify(options.points));
	if (text) form.append("text", text);
	form.append("image_key", options.imageKey);

	const response = await fetch(SAM3_SEGMENT_ENDPOINT, {
		method: "POST",
		body: form,
		signal: options.signal,
	});
	if (!response.ok) {
		throw new SamApiError(response.status, await readDetail(response));
	}
	const payload = (await response.json()) as {
		rle: RawRle;
		runs: DecodedMask["runs"];
		height: number;
		width: number;
		score: number;
		area: number;
		bbox: [number, number, number, number] | null;
		embedding_reused: boolean;
		encoder_ms: number;
		decoder_ms: number;
		instances: number;
		instance_scores: number[];
		exemplars: number[][];
	};
	return {
		rle: payload.rle,
		mask: { height: payload.height, width: payload.width, runs: payload.runs },
		score: payload.score,
		area: payload.area,
		bbox: payload.bbox,
		embeddingReused: payload.embedding_reused,
		encoderMs: payload.encoder_ms,
		decoderMs: payload.decoder_ms,
		instances: payload.instances,
		instanceScores: payload.instance_scores ?? [],
		exemplars: payload.exemplars ?? [],
	};
}

export interface PropagateStatus {
	sam2: SamStatus;
	sam3: SamStatus;
	/** Maximum frames per request, anchor included (PROPAGATE_MAX_FRAMES). */
	maxFrames: number;
}

export type PropagateBackend = "sam2" | "sam3";

export async function fetchPropagateStatus(
	signal?: AbortSignal,
): Promise<PropagateStatus> {
	const fallback = (error: string): PropagateStatus => ({
		sam2: unavailable("SAM 2", error),
		sam3: unavailable("SAM 3", error),
		maxFrames: 0,
	});
	try {
		const response = await fetch(PROPAGATE_STATUS_ENDPOINT, { signal });
		if (!response.ok) {
			return fallback(
				`Backend answered ${response.status} for the propagate status.`,
			);
		}
		const payload = (await response.json()) as {
			sam2?: Partial<SamStatus>;
			sam3?: Partial<SamStatus>;
			max_frames?: number;
		};
		const read = (part: Partial<SamStatus> | undefined, name: string) => ({
			available: Boolean(part?.available),
			loaded: Boolean(part?.loaded),
			model: part?.model ?? name,
			device: part?.device ?? "?",
			error: part?.error ?? null,
		});
		return {
			sam2: read(payload.sam2, "SAM 2"),
			sam3: read(payload.sam3, "SAM 3"),
			maxFrames: payload.max_frames ?? 0,
		};
	} catch (cause) {
		if (cause instanceof DOMException && cause.name === "AbortError")
			throw cause;
		return fallback(
			"Backend is not reachable; start it with `uvicorn main:app`.",
		);
	}
}

export interface PropagateFrame {
	/** Clip frame index. */
	index: number;
	/** The frame image straight out of the archive. */
	image: Blob;
}

export interface PropagateOptions {
	/** Every frame of the window, anchor included. */
	frames: PropagateFrame[];
	/** Frame index that carries the selected mask. */
	anchor: number;
	/** The selected mask on the anchor frame. */
	mask: RawRle;
	/** Frames to track before / after the anchor. */
	backward: number;
	forward: number;
	backend: PropagateBackend;
	signal?: AbortSignal;
}

export interface PropagatedFrame {
	frameIndex: number;
	rle: RawRle;
	mask: DecodedMask;
	area: number;
}

export interface PropagateResult {
	backend: PropagateBackend;
	model: string;
	device: string;
	elapsedMs: number;
	masks: PropagatedFrame[];
}

/**
 * Propagate one mask over its neighbouring frames. 
 */
export async function propagateMask(
	options: PropagateOptions,
): Promise<PropagateResult> {
	if (options.backward <= 0 && options.forward <= 0) {
		throw new SamApiError(0, "Choose at least one frame to propagate to.");
	}
	const form = new FormData();
	for (const frame of options.frames) {
		form.append("frames", frame.image, `${frame.index}.jpg`);
	}
	form.append(
		"frame_indices",
		JSON.stringify(options.frames.map((frame) => frame.index)),
	);
	form.append("anchor", String(options.anchor));
	form.append("mask", JSON.stringify(options.mask));
	form.append("backward", String(options.backward));
	form.append("forward", String(options.forward));
	form.append("backend", options.backend);

	const response = await fetch(PROPAGATE_ENDPOINT, {
		method: "POST",
		body: form,
		signal: options.signal,
	});
	if (!response.ok) {
		throw new SamApiError(response.status, await readDetail(response));
	}
	const payload = (await response.json()) as {
		backend: PropagateBackend;
		model: string;
		device: string;
		height: number;
		width: number;
		elapsed_ms: number;
		masks: {
			frame_index: number;
			rle: RawRle;
			runs: DecodedMask["runs"];
			area: number;
		}[];
	};
	return {
		backend: payload.backend,
		model: payload.model,
		device: payload.device,
		elapsedMs: payload.elapsed_ms,
		masks: payload.masks.map((item) => ({
			frameIndex: item.frame_index,
			rle: item.rle,
			mask: { height: payload.height, width: payload.width, runs: item.runs },
			area: item.area,
		})),
	};
}

async function readDetail(response: Response): Promise<string> {
	try {
		const body = (await response.json()) as { detail?: unknown };
		if (typeof body.detail === "string") return body.detail;
		if (body.detail) return JSON.stringify(body.detail);
	} catch {
		/* not JSON */
	}
	return `Segmentation request failed (${response.status}).`;
}
