import { API_BASE } from "./apiBase";

/**
 * Client for `/api/sessions`.
 *
 * Opening a project hands the archive to the backend, which materialises the
 * frames into a session and reports the sequence back. Everything on screen
 * comes from that session, so the reviewer never has to decode a container
 * itself.
 */

const SESSIONS_ENDPOINT = `${API_BASE}/api/sessions`;

/** What the backend made of an uploaded archive. */
export interface OpenedProject {
    sessionId: string;
    /** `"frames"` when the archive carried them, `"video"` when they were decoded. */
    source: "frames" | "video";
    frameNames: string[];
    frameCount: number;
    fps: number;
    width: number;
    height: number;
    mode: string | null;
    originalFps: number | null;
    videoEntry: string | null;
    archive: string | null;
    recordedFrameNames: string[];
    /** `false` when the archive's own frame list disagreed with the frames built. */
    frameNamesMatch: boolean;
    expiresInSeconds: number;
}

export class SessionApiError extends Error {
    readonly status: number;

    constructor(status: number, message: string) {
        super(message);
        this.name = "SessionApiError";
        this.status = status;
    }
}

async function errorDetail(response: Response): Promise<string> {
    try {
        const payload = (await response.json()) as { detail?: unknown };
        if (typeof payload?.detail === "string") return payload.detail;
    } catch {
        /* not JSON; fall through to the status text */
    }
    return response.statusText || `The backend answered ${response.status}.`;
}

/** Upload an archive and wait for the backend to materialise its frames. */
export async function openProject(
    file: File,
    signal?: AbortSignal,
): Promise<OpenedProject> {
    const form = new FormData();
    form.append("project", file, file.name || "project.project");

    let response: Response;
    try {
        response = await fetch(SESSIONS_ENDPOINT, {
            method: "POST",
            body: form,
            signal,
        });
    } catch (cause) {
        if (cause instanceof DOMException && cause.name === "AbortError")
            throw cause;
        throw new SessionApiError(
            0,
            "The backend is not reachable, so the frames could not be prepared. " +
                "Start it with `uvicorn src.main:app --port 8000` from `backend/`.",
        );
    }

    if (!response.ok) {
        throw new SessionApiError(response.status, await errorDetail(response));
    }

    const payload = (await response.json()) as {
        session_id: string;
        source: string;
        frame_names: string[];
        frame_count: number;
        fps: number;
        width: number;
        height: number;
        mode: string | null;
        original_fps: number | null;
        video_entry: string | null;
        archive: string | null;
        recorded_frame_names: string[];
        frame_names_match: boolean;
        expires_in_seconds: number;
    };

    return {
        sessionId: payload.session_id,
        source: payload.source === "video" ? "video" : "frames",
        frameNames: payload.frame_names ?? [],
        frameCount: payload.frame_count ?? 0,
        fps: payload.fps ?? 0,
        width: payload.width ?? 0,
        height: payload.height ?? 0,
        mode: payload.mode ?? null,
        originalFps: payload.original_fps ?? null,
        videoEntry: payload.video_entry ?? null,
        archive: payload.archive ?? null,
        recordedFrameNames: payload.recorded_frame_names ?? [],
        frameNamesMatch: payload.frame_names_match ?? true,
        expiresInSeconds: payload.expires_in_seconds ?? 0,
    };
}

/**
 * Drop a session. Best-effort and non-blocking: this is called while the page
 * is going away, so `keepalive` is what lets the request outlive the page.
 */
export function closeProject(sessionId: string): void {
    void fetch(`${SESSIONS_ENDPOINT}/${encodeURIComponent(sessionId)}`, {
        method: "DELETE",
        keepalive: true,
    }).catch(() => {
        /* the sweeper will reap it instead */
    });
}
