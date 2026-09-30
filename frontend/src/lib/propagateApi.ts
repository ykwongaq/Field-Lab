import type { RawRle } from "../types";
import { API_BASE, apiFetch } from "./apiBase";

/**
 * Client for mask propagation jobs.
 *
 * A run can cover thousands of frames, so it is a background job rather than a
 * request: start it, then poll. `since` on a poll returns only the frames newer
 * than the highest one already held, which is what lets the reviewer paint masks
 * as they arrive without re-downloading the run on every poll.
 *
 * One job propagates one mask; several objects are queued one after another by
 * the backend, which is what keeps GPU memory bounded.
 */

export type PropagationDirection = "forward" | "backward" | "both";

export type JobState = "queued" | "running" | "done" | "cancelled" | "failed";

export interface PropagateStatus {
    available: boolean;
    loaded: boolean;
    model: string;
    device: string;
    error: string | null;
    /** Frames one session holds at a time. */
    windowFrames: number;
    /** Frames consecutive windows share (the re-anchoring stretch). */
    overlap: number;
    /**
     * What a window with no verified frame may be seeded with. `derived` hands
     * it the previous window's own output; `verified` stops the run instead.
     */
    chaining: string;
}

export interface JobProgress {
    framesDone: number;
    /** Frames the run must produce; the anchor is not one of them. */
    framesTotal: number;
    windowIndex: number;
    windowsTotal: number;
}

export interface PropagatedFrame {
    frameIndex: number;
    rle: RawRle;
    area: number;
}

export interface PropagationJob {
    jobId: string;
    sessionId: string;
    state: JobState;
    error: string | null;
    anchor: number;
    direction: PropagationDirection;
    first: number;
    last: number;
    progress: JobProgress;
    plan: {
        frames_total?: number;
        frames_to_produce?: number;
        windows_total?: number;
        window_frames?: number;
        elapsed_ms?: number;
        /** How many verified frames the run was seeded with (anchor excluded). */
        pinned?: number;
        /**
         * The windows the run was split into, in order. Each one is its own SAM 3
         * session, and the boundary between two of them is where a track can
         * drift, so the timeline draws a tick there.
         */
        windows?: { start: number; end: number; direction?: string }[];
    } | null;
    /** How many jobs were already waiting when this one was submitted. */
    queuePosition: number;
    masks: PropagatedFrame[];
}

export interface StartPropagationOptions {
    sessionId: string;
    anchorFrame: number;
    mask: RawRle;
    direction?: PropagationDirection;
    /** Inclusive frame range; omit both for the whole clip. */
    first?: number;
    last?: number;
    objectId?: number;
    /**
     * Frames a human verified by hand, other than the anchor.
     *
     * They are written into tracker memory as authoritative exactly like the
     * anchor, which is what makes a re-propagate restart from the correction
     * rather than from the run it is replacing. Model output must never be sent
     * here: a window conditioned on its own previous guess keeps its mistakes.
     */
    pins?: Map<number, RawRle>;
    /** `verified` refuses to seed an unverified window from a prediction. */
    chaining?: "derived" | "verified";
    signal?: AbortSignal;
}

export class PropagateApiError extends Error {
    readonly status: number;

    constructor(status: number, message: string) {
        super(message);
        this.name = "PropagateApiError";
        this.status = status;
    }
}

const STATUS_ENDPOINT = `${API_BASE}/api/propagate/status`;
const JOBS_ENDPOINT = `${API_BASE}/api/propagate/jobs`;

interface WireJob {
    job_id: string;
    session_id: string;
    state: string;
    error: string | null;
    anchor: number;
    direction: string;
    first: number;
    last: number;
    progress: {
        frames_done: number;
        frames_total: number;
        window_index: number;
        windows_total: number;
    };
    plan: Record<string, unknown> | null;
    queue_position: number;
    masks: {
        frame_index: number;
        rle: { size: [number, number]; counts: string | number[] };
        area: number;
    }[];
}

/** Tracker availability plus how the run will be windowed. */
export async function fetchPropagateStatus(
    signal?: AbortSignal,
): Promise<PropagateStatus> {
    const fallback = (error: string): PropagateStatus => ({
        available: false,
        loaded: false,
        model: "SAM 3 tracker",
        device: "?",
        error,
        windowFrames: 0,
        overlap: 0,
        chaining: "derived",
    });
    try {
        const response = await apiFetch(STATUS_ENDPOINT, { signal });
        if (!response.ok) {
            return fallback(
                `Backend answered ${response.status} for the tracker status.`,
            );
        }
        const payload = (await response.json()) as {
            sam3: {
                available: boolean;
                loaded: boolean;
                model: string;
                device: string;
                error: string | null;
            };
            window_frames: number;
            overlap: number;
            chaining?: string;
        };
        return {
            available: Boolean(payload.sam3.available),
            loaded: Boolean(payload.sam3.loaded),
            model: payload.sam3.model,
            device: payload.sam3.device,
            error: payload.sam3.error ?? null,
            windowFrames: payload.window_frames,
            overlap: payload.overlap,
            chaining: payload.chaining ?? "derived",
        };
    } catch (cause) {
        if (cause instanceof DOMException && cause.name === "AbortError")
            throw cause;
        return fallback("Backend is not reachable; the tracker cannot run.");
    }
}

function toJob(payload: WireJob): PropagationJob {
    return {
        jobId: payload.job_id,
        sessionId: payload.session_id,
        state: payload.state as JobState,
        error: payload.error,
        anchor: payload.anchor,
        direction: payload.direction as PropagationDirection,
        first: payload.first,
        last: payload.last,
        progress: {
            framesDone: payload.progress.frames_done,
            framesTotal: payload.progress.frames_total,
            windowIndex: payload.progress.window_index,
            windowsTotal: payload.progress.windows_total,
        },
        plan: payload.plan as PropagationJob["plan"],
        queuePosition: payload.queue_position,
        masks: payload.masks.map((item) => ({
            frameIndex: item.frame_index,
            rle: item.rle,
            area: item.area,
        })),
    };
}

/** Queue one propagation run and return it immediately. */
export async function startPropagation(
    options: StartPropagationOptions,
): Promise<PropagationJob> {
    const body: Record<string, unknown> = {
        session_id: options.sessionId,
        anchor_frame: options.anchorFrame,
        mask: options.mask,
        direction: options.direction ?? "both",
    };
    if (options.first !== undefined) body.first = options.first;
    if (options.last !== undefined) body.last = options.last;
    if (options.objectId !== undefined) body.object_id = options.objectId;
    if (options.pins && options.pins.size > 0) {
        // Ascending order so the request (and any log of it) is stable.
        body.pins = [...options.pins.entries()]
            .sort(([left], [right]) => left - right)
            .map(([frameIndex, mask]) => ({
                frame_index: frameIndex,
                mask,
            }));
    }
    if (options.chaining) body.chaining = options.chaining;

    const response = await apiFetch(JOBS_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: options.signal,
    });
    if (!response.ok) {
        throw new PropagateApiError(
            response.status,
            await readDetail(response),
        );
    }
    return toJob((await response.json()) as WireJob);
}

/**
 * Poll one job.
 *
 * `since` is the highest frame index already held; only newer masks come back,
 * so a poll costs the same whether the run is 50 frames in or 5,000.
 */
export async function getPropagationJob(
    jobId: string,
    options: {
        since?: number;
        includeMasks?: boolean;
        signal?: AbortSignal;
    } = {},
): Promise<PropagationJob> {
    const query = new URLSearchParams();
    if (options.since !== undefined) query.set("since", String(options.since));
    if (options.includeMasks === false) query.set("include_masks", "false");
    const suffix = query.toString() ? `?${query}` : "";
    const response = await apiFetch(
        `${JOBS_ENDPOINT}/${encodeURIComponent(jobId)}${suffix}`,
        { signal: options.signal },
    );
    if (!response.ok) {
        throw new PropagateApiError(
            response.status,
            await readDetail(response),
        );
    }
    return toJob((await response.json()) as WireJob);
}

/** Ask a job to stop. Whatever it produced before that is kept. */
export async function cancelPropagation(
    jobId: string,
): Promise<PropagationJob> {
    const response = await apiFetch(
        `${JOBS_ENDPOINT}/${encodeURIComponent(jobId)}`,
        { method: "DELETE" },
    );
    if (!response.ok) {
        throw new PropagateApiError(
            response.status,
            await readDetail(response),
        );
    }
    return toJob((await response.json()) as WireJob);
}

/** Whether a job is still occupying the queue. */
export function jobIsActive(job: PropagationJob): boolean {
    return job.state === "queued" || job.state === "running";
}

async function readDetail(response: Response): Promise<string> {
    try {
        const payload = (await response.json()) as { detail?: unknown };
        if (typeof payload.detail === "string") return payload.detail;
        if (Array.isArray(payload.detail) && payload.detail.length > 0) {
            const first = payload.detail[0] as { msg?: string };
            if (first?.msg) return first.msg;
        }
    } catch {
        // fall through
    }
    return `The propagation request failed (${response.status}).`;
}
