/**
 * Read a video's basic facts in the browser.
 *
 * Duration and dimensions come from the media metadata. The frame rate is *not*
 * exposed by HTML, so it is measured: `requestVideoFrameCallback` reports the
 * media time of each presented frame, and the spacing between two of them is the
 * frame period. The element is rendered (but invisible and 1px) because a
 * `display: none` video is not decoded, so no frames would ever be presented.
 *
 * Everything here is best-effort: the probe always resolves, and `fps` is simply
 * `null` when the browser cannot measure it (the reviewer then types the rate).
 */

export interface VideoInfo {
    durationSeconds: number;
    width: number;
    height: number;
    /** Measured frames per second, or null when it could not be measured. */
    fps: number | null;
}

const MAX_PROBE_MS = 4000;
const WANTED_FRAMES = 12;
/** Below this the estimate is too noisy to show. */
const MIN_FRAMES = 4;
const MIN_SPAN_SECONDS = 0.05;

interface FrameMetadata {
    mediaTime: number;
    presentedFrames?: number;
}

/** Ask the element for the next presented frame, if it supports it. */
function nextFrame(
    video: HTMLVideoElement,
): Promise<FrameMetadata | null> | null {
    const request = video.requestVideoFrameCallback?.bind(video);
    if (!request) return null;
    return new Promise((resolve) => {
        request((_now, metadata) => resolve(metadata));
    });
}

export async function probeVideo(file: File): Promise<VideoInfo> {
    const url = URL.createObjectURL(file);
    const video = document.createElement("video");
    video.src = url;
    video.muted = true;
    video.defaultMuted = true;
    video.playsInline = true;
    video.preload = "metadata";
    // Rendered but invisible: enough to decode, invisible to the user.
    Object.assign(video.style, {
        position: "fixed",
        top: "0",
        left: "0",
        width: "1px",
        height: "1px",
        opacity: "0",
        pointerEvents: "none",
    });
    document.body.appendChild(video);

    try {
        await waitForMetadata(video);
        const info: VideoInfo = {
            durationSeconds: Number.isFinite(video.duration)
                ? video.duration
                : 0,
            width: video.videoWidth,
            height: video.videoHeight,
            fps: await measureFps(video),
        };
        return info;
    } catch {
        return { durationSeconds: 0, width: 0, height: 0, fps: null };
    } finally {
        video.pause();
        video.removeAttribute("src");
        video.remove();
        URL.revokeObjectURL(url);
    }
}

function waitForMetadata(video: HTMLVideoElement): Promise<void> {
    if (video.readyState >= HTMLMediaElement.HAVE_METADATA)
        return Promise.resolve();
    return new Promise((resolve, reject) => {
        const timer = window.setTimeout(
            () => reject(new Error("Timed out reading the video metadata")),
            MAX_PROBE_MS,
        );
        video.addEventListener(
            "loadedmetadata",
            () => {
                window.clearTimeout(timer);
                resolve();
            },
            { once: true },
        );
        video.addEventListener(
            "error",
            () => {
                window.clearTimeout(timer);
                reject(new Error("The video could not be read"));
            },
            { once: true },
        );
    });
}

async function measureFps(video: HTMLVideoElement): Promise<number | null> {
    if (!video.requestVideoFrameCallback) return null;

    const deadline = performance.now() + MAX_PROBE_MS;
    const times: number[] = [];
    try {
        await video.play();
    } catch {
        return null; // autoplay refused; the reviewer types the rate instead
    }
    try {
        while (times.length < WANTED_FRAMES && performance.now() < deadline) {
            const pending = nextFrame(video);
            if (!pending) return null;
            const metadata = await pending;
            if (!metadata) break;
            times.push(metadata.mediaTime);
        }
    } finally {
        video.pause();
        video.currentTime = 0;
    }

    if (times.length < MIN_FRAMES) return null;
    const span = times[times.length - 1] - times[0];
    if (span < MIN_SPAN_SECONDS) return null;
    const fps = (times.length - 1) / span;
    if (!Number.isFinite(fps) || fps <= 0) return null;
    return Math.round(fps * 1000) / 1000;
}

/** `1:23.4` for a duration in seconds. */
export function formatDuration(seconds: number): string {
    if (!Number.isFinite(seconds) || seconds <= 0) return "–";
    const total = Math.round(seconds * 10) / 10;
    const minutes = Math.floor(total / 60);
    const rest = total - minutes * 60;
    return minutes > 0
        ? `${minutes}:${rest.toFixed(1).padStart(4, "0")}`
        : `${rest.toFixed(1)}s`;
}
