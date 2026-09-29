import { API_BASE } from "./apiBase";

/**
 * Where a clip's frame pixels come from.
 *
 * The reviewer always displays the frames the backend materialised, so a project
 * plays the same whether its archive carried a frame folder or a video. Keeping
 * that behind an interface means the display path never has to know which of the
 * two it was, and the archive is left to hold the annotation.
 */
export interface FrameSource {
    /** Frame names in playback order, as the backend reported them. */
    readonly names: readonly string[];
    readonly count: number;
    /** One frame, as bytes the browser can decode. */
    frame(index: number): Promise<Blob>;
}

/** Frames the backend extracted into a session. */
export class SessionFrameSource implements FrameSource {
    readonly sessionId: string;
    readonly names: readonly string[];

    constructor(sessionId: string, names: readonly string[]) {
        this.sessionId = sessionId;
        this.names = names;
    }

    get count(): number {
        return this.names.length;
    }

    /** Where frame `index` is served from. */
    frameUrl(index: number): string {
        return `${API_BASE}/api/sessions/${encodeURIComponent(
            this.sessionId,
        )}/frames/${index}`;
    }

    async frame(index: number): Promise<Blob> {
        if (!Number.isInteger(index) || index < 0 || index >= this.count) {
            throw new Error(
                `Frame ${index} is outside this clip (0–${this.count - 1}).`,
            );
        }
        const response = await fetch(this.frameUrl(index));
        if (!response.ok) {
            throw new Error(
                `Frame ${index} could not be read from the backend ` +
                    `(${response.status}). The session may have expired — ` +
                    "reopen the project to build a new one.",
            );
        }
        return response.blob();
    }
}
