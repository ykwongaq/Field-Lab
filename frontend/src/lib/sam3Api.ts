import type { DecodedMask, PromptPoint, RawRle } from "../types";
import { API_BASE } from "./apiBase";

/**
 * Client for SAM 3 prompts on one frame of an open session.
 *
 * The frame is named, not uploaded: the backend already holds the clip's pixels
 * in the session, so a click costs one small JSON request instead of an image
 * upload, and the frame's vision embeddings stay warm for the whole
 * click-by-click refinement of a mask.
 */

export interface Sam3Status {
    available: boolean;
    loaded: boolean;
    model: string;
    device: string;
    /** Detection threshold applied to text prompts. */
    threshold: number;
    /** Which models are currently resident ("image", "video"). */
    loadedModels: string[];
    /** False when point and box prompts are unavailable. */
    pointPrompts: boolean;
    error: string | null;
}

/** What the user is asking for. */
export type PromptKind = "point" | "box" | "text";

export interface PromptBox {
    x0: number;
    y0: number;
    x1: number;
    y1: number;
    /** 1 = include, 0 = exclude. */
    label?: 0 | 1;
}

export interface SegmentOptions {
    sessionId: string;
    frameIndex: number;
    kind: PromptKind;
    points?: PromptPoint[];
    boxes?: PromptBox[];
    text?: string;
    /** 0 (the default) keeps every proposal. */
    maxInstances?: number;
    signal?: AbortSignal;
}

/** One proposed mask, so the reviewer can offer the alternatives. */
export interface MaskInstance {
    rle: RawRle;
    mask: DecodedMask;
    score: number;
    area: number;
    bbox: [number, number, number, number] | null;
}

export interface SegmentResult {
    /** The mask to use: the best candidate, or the union for a text prompt. */
    rle: RawRle;
    mask: DecodedMask;
    score: number;
    area: number;
    bbox: [number, number, number, number] | null;
    kind: PromptKind;
    prompt: string;
    instances: MaskInstance[];
    instanceScores: number[];
    embeddingReused: boolean;
    encoderMs: number;
    decoderMs: number;
}

export class Sam3ApiError extends Error {
    readonly status: number;

    constructor(status: number, message: string) {
        super(message);
        this.name = "Sam3ApiError";
        this.status = status;
    }
}

const STATUS_ENDPOINT = `${API_BASE}/api/sam3/status`;
const SEGMENT_ENDPOINT = `${API_BASE}/api/sam3/segment`;

interface WireRle {
    size: [number, number];
    counts: string | number[];
}

interface WireInstance {
    rle: WireRle;
    runs: { x: number; y: number; length: number }[];
    score: number;
    area: number;
    bbox: [number, number, number, number] | null;
}

interface WireSegmentResponse {
    rle: WireRle;
    runs: { x: number; y: number; length: number }[];
    height: number;
    width: number;
    score: number;
    area: number;
    bbox: [number, number, number, number] | null;
    kind: PromptKind;
    prompt: string;
    embedding_reused: boolean;
    encoder_ms: number;
    decoder_ms: number;
    instances: WireInstance[];
    instance_scores: number[];
}

/** SAM 3 availability and what prompting it supports. */
export async function fetchSam3Status(
    signal?: AbortSignal,
): Promise<Sam3Status> {
    try {
        const response = await fetch(STATUS_ENDPOINT, { signal });
        if (!response.ok) {
            return unavailable(
                `Backend answered ${response.status} for the SAM 3 status.`,
            );
        }
        const payload = (await response.json()) as Partial<Sam3Status> & {
            loaded_models?: string[];
            point_prompts?: boolean;
        };
        return {
            available: Boolean(payload.available),
            loaded: Boolean(payload.loaded),
            model: payload.model ?? "SAM 3",
            device: payload.device ?? "?",
            threshold: payload.threshold ?? 0.5,
            loadedModels: payload.loadedModels ?? payload.loaded_models ?? [],
            pointPrompts: payload.pointPrompts ?? payload.point_prompts ?? true,
            error: payload.error ?? null,
        };
    } catch (cause) {
        if (cause instanceof DOMException && cause.name === "AbortError")
            throw cause;
        return unavailable(
            "Backend is not reachable; start it with `uvicorn src.main:app`.",
        );
    }
}

function unavailable(error: string): Sam3Status {
    return {
        available: false,
        loaded: false,
        model: "SAM 3",
        device: "?",
        threshold: 0.5,
        loadedModels: [],
        pointPrompts: false,
        error,
    };
}

function toDecoded(
    height: number,
    width: number,
    runs: { x: number; y: number; length: number }[],
): DecodedMask {
    return { height, width, runs };
}

/**
 * Turn one prompt into a mask.
 *
 * `point` and `box` mean *this object*; `text` means *this class* and may match
 * several objects, which come back in `instances` so an instance project can
 * split them instead of merging them into one tracklet.
 */
export async function segmentFrame(
    options: SegmentOptions,
): Promise<SegmentResult> {
    const body: Record<string, unknown> = {
        session_id: options.sessionId,
        frame_index: options.frameIndex,
        prompt: {
            kind: options.kind,
            points: options.points ?? [],
            boxes: options.boxes ?? [],
            text: options.text ?? null,
        },
    };
    if (options.maxInstances) body.max_instances = options.maxInstances;

    const response = await fetch(SEGMENT_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: options.signal,
    });
    if (!response.ok) {
        throw new Sam3ApiError(response.status, await readDetail(response));
    }
    const payload = (await response.json()) as WireSegmentResponse;
    return {
        rle: payload.rle,
        mask: toDecoded(payload.height, payload.width, payload.runs),
        score: payload.score,
        area: payload.area,
        bbox: payload.bbox,
        kind: payload.kind,
        prompt: payload.prompt,
        instances: payload.instances.map((item) => ({
            rle: item.rle,
            mask: toDecoded(payload.height, payload.width, item.runs),
            score: item.score,
            area: item.area,
            bbox: item.bbox,
        })),
        instanceScores: payload.instance_scores,
        embeddingReused: payload.embedding_reused,
        encoderMs: payload.encoder_ms,
        decoderMs: payload.decoder_ms,
    };
}

/** Pull `detail` out of an error response, whatever shape it has. */
async function readDetail(response: Response): Promise<string> {
    try {
        const payload = (await response.json()) as { detail?: unknown };
        if (typeof payload.detail === "string") return payload.detail;
        if (Array.isArray(payload.detail) && payload.detail.length > 0) {
            const first = payload.detail[0] as { msg?: string };
            if (first?.msg) return first.msg;
        }
    } catch {
        // fall through to the generic message
    }
    return `The segmentation request failed (${response.status}).`;
}
