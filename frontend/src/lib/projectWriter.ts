/**
 * Build a `.project` archive in the browser.
 *
 * Creation is deliberately dumb: whatever the user provides is stored verbatim
 * (a source video, or a folder of frames) next to the annotation JSON and a
 * metadata JSON. Nothing is decoded, resampled or re-encoded, so a multi-gigabyte
 * video can be packed without a server round-trip.
 *
 * Archive layout (the contract with `lib/clip.ts`):
 *
 *     video/<name><ext>          when a video was provided
 *     frames/<name>              when frames were provided (order = natural sort)
 *     annotations/<name>.json    VideoSegmentation dataset
 *     metadata.json              free-form user metadata
 */

import type { RawDataset, RawVideo } from "../types";
import type { ProjectMode } from "./project";
import { ZipWriter } from "./zipWriter";

export const PROJECT_EXTENSION = ".project";
export const METADATA_ENTRY = "metadata.json";

const ANNOTATIONS_DIR = "annotations";
const FRAMES_DIR = "frames";
const VIDEO_DIR = "video";
const DEFAULT_FPS_FALLBACK = 25;

const SAFE_NAME = /[^A-Za-z0-9._-]+/g;

export type CreateProjectProgress =
    | { stage: "frames"; done: number; total: number }
    | { stage: "video"; loaded: number; total: number };

export interface CreateProjectInput {
    /** Project name: used for the archive, the annotation entry and video_name. */
    name: string;
    mode: ProjectMode;
    /** Source video, when the user gave one. */
    video?: File | null;
    /** Frames of one clip, when the user gave those instead. */
    frames?: readonly File[] | null;
    /** Optional VideoSegmentation JSON that seeds the annotations. */
    annotation?: File | null;
    /** Free-form metadata, stored verbatim as `metadata.json`. */
    metadata?: Record<string, unknown> | null;
    /** Frame rate of the source material, recorded for provenance. */
    originalFps?: number | null;
    /** Frame rate the reviewer should treat the project as having. */
    targetFps?: number | null;
    onProgress?: (progress: CreateProjectProgress) => void;
    signal?: AbortSignal;
}

export interface CreatedProject {
    /** The finished archive, ready to be saved and opened in the reviewer. */
    blob: Blob;
    fileName: string;
    name: string;
    mode: ProjectMode;
    frameCount: number;
    fps: number;
    /** What was stored: a source video, or the frames themselves. */
    source: "video" | "frames";
    frameNames: string[];
}

/** Strip anything a file name should not carry, mirroring the backend. */
export function sanitizeName(raw: string): string {
    return raw.replace(SAFE_NAME, "_").replace(/^[._-]+|[._-]+$/g, "");
}

function basename(name: string): string {
    return name.replace(/\\/g, "/").split("/").pop() ?? name;
}

function stem(name: string): string {
    return basename(name).replace(/\.[^.]+$/, "");
}

function extensionOf(name: string): string {
    const match = /\.[^./\\]+$/.exec(basename(name));
    return match ? match[0].toLowerCase() : ".mp4";
}

/** Natural order, so `frame_2.jpg` sorts before `frame_10.jpg`. */
export function naturalCompare(a: string, b: string): number {
    const left = a.split(/(\d+)/);
    const right = b.split(/(\d+)/);
    const shared = Math.min(left.length, right.length);
    for (let i = 0; i < shared; i++) {
        const x = left[i];
        const y = right[i];
        if (x === y) continue;
        const numeric = /^\d+$/.test(x) && /^\d+$/.test(y);
        if (numeric) {
            const difference = Number(x) - Number(y);
            if (difference !== 0) return difference;
            continue;
        }
        return x < y ? -1 : 1;
    }
    return left.length - right.length;
}

/** Order the frames and give every entry a unique name inside `frames/`. */
function orderFrames(frames: readonly File[]): { file: File; name: string }[] {
    const used = new Set<string>();
    return frames
        .map((file) => ({ file, name: basename(file.name) }))
        .sort((a, b) => naturalCompare(a.name, b.name))
        .map((entry) => ({
            file: entry.file,
            name: uniqueName(entry.name, used),
        }));
}

function uniqueName(name: string, used: Set<string>): string {
    if (!used.has(name)) {
        used.add(name);
        return name;
    }
    const dot = name.lastIndexOf(".");
    const base = dot > 0 ? name.slice(0, dot) : name;
    const extension = dot > 0 ? name.slice(dot) : "";
    for (let index = 1; ; index++) {
        const candidate = `${base}~${index}${extension}`;
        if (!used.has(candidate)) {
            used.add(candidate);
            return candidate;
        }
    }
}

/** Frame size, read from one frame so masks can be scaled to the right canvas. */
async function frameSize(
    file: File,
): Promise<{ width: number; height: number }> {
    try {
        const bitmap = await createImageBitmap(file);
        const size = { width: bitmap.width, height: bitmap.height };
        bitmap.close();
        return size;
    } catch {
        // A format this browser cannot decode is still worth storing; the
        // reviewer will report the frames it cannot draw.
        return { width: 0, height: 0 };
    }
}

async function readAnnotationFile(file: File): Promise<RawDataset> {
    let parsed: unknown;
    try {
        parsed = JSON.parse(await file.text());
    } catch (cause) {
        throw new Error(
            `The annotation file is not valid JSON: ${
                cause instanceof Error ? cause.message : String(cause)
            }`,
        );
    }
    if (
        typeof parsed !== "object" ||
        parsed === null ||
        Array.isArray(parsed)
    ) {
        throw new Error("The annotation file must contain a JSON object.");
    }

    const dataset = parsed as RawDataset;
    const videos = dataset.videos;
    if (videos !== undefined) {
        if (!Array.isArray(videos)) {
            throw new Error("`videos` in the annotation file must be a list.");
        }
        if (videos.length > 1) {
            throw new Error(
                `The annotation file describes ${videos.length} videos; a project holds exactly one clip.`,
            );
        }
    } else if (
        dataset.annotations === undefined &&
        dataset.categories === undefined
    ) {
        throw new Error(
            "The annotation file does not look like a VideoSegmentation dataset " +
                "(expected `videos`, `annotations` and `categories`).",
        );
    }
    if (
        dataset.annotations !== undefined &&
        !Array.isArray(dataset.annotations)
    ) {
        throw new Error("`annotations` in the annotation file must be a list.");
    }
    if (
        dataset.categories !== undefined &&
        !Array.isArray(dataset.categories)
    ) {
        throw new Error("`categories` in the annotation file must be a list.");
    }
    return dataset;
}

/**
 * Complete the dataset so it describes the frames actually stored.
 *
 * `annotations` and `categories`, plus any extra keys on the video record (say
 * `scene_id`), are kept as the user supplied them. The fields that describe the
 * frames are overwritten — they have to agree with the archive — while `id` and
 * `status` survive, because annotation `video_id`s point at the former and the
 * latter records how far the review got.
 */
function withVideoRecord(
    dataset: RawDataset | null,
    record: RawVideo,
): RawDataset {
    if (!dataset) {
        return { videos: [record], annotations: [], categories: [] };
    }
    const videos = dataset.videos ?? [];
    if (videos.length === 0) {
        dataset.videos = [record];
    } else {
        const existing = videos[0];
        const preserved: Partial<RawVideo> = {};
        if (existing.id !== undefined && existing.id !== null) {
            preserved.id = existing.id;
        }
        if (existing.status) preserved.status = existing.status;
        dataset.videos = [{ ...existing, ...record, ...preserved }];
    }
    dataset.annotations ??= [];
    dataset.categories ??= [];
    return dataset;
}

function throwIfAborted(signal: AbortSignal | undefined): void {
    if (signal?.aborted) {
        throw new DOMException("Project creation aborted", "AbortError");
    }
}

/**
 * Pack a project archive from what the user provided.
 *
 * At least one of `video` or `frames` is needed, and both is fine: the frames
 * are what the reviewer looks at, the video keeps the provenance. `originalFps`
 * / `targetFps` are recorded in the annotation file as intent — they never
 * change the stored bytes, so no frame rate maths happens here.
 */
export async function createProjectFile(
    input: CreateProjectInput,
): Promise<CreatedProject> {
    const frames = input.frames ? [...input.frames] : [];
    const video = input.video ?? null;
    const hasVideo = video !== null;
    const hasFrames = frames.length > 0;
    if (!hasVideo && !hasFrames) {
        throw new Error("Provide a video, a set of frames, or both.");
    }

    const name =
        sanitizeName(input.name) || (video ? stem(video.name) : "project");
    const metadata = input.metadata ?? {};
    try {
        JSON.stringify(metadata);
    } catch (cause) {
        throw new Error(
            `The metadata is not JSON-serialisable: ${
                cause instanceof Error ? cause.message : String(cause)
            }`,
        );
    }

    const ordered = orderFrames(frames);
    const frameNames = ordered.map((entry) => entry.name);
    const size = ordered.length
        ? await frameSize(ordered[0].file)
        : { width: 0, height: 0 };
    const fps = input.targetFps ?? input.originalFps ?? DEFAULT_FPS_FALLBACK;

    const videoEntry = video
        ? `${VIDEO_DIR}/${name}${extensionOf(video.name)}`
        : null;
    const record: RawVideo = {
        id: 1,
        video_name: name,
        file_names: frameNames,
        length: frameNames.length,
        height: size.height,
        width: size.width,
        fps,
        original_video: video ? basename(video.name) : null,
        segmentation_mode: input.mode,
        video_file: videoEntry,
        original_fps: input.originalFps ?? null,
        target_fps: input.targetFps ?? null,
        frame_step: 1,
        start_frame: 0,
        end_frame: Math.max(0, frameNames.length - 1),
        status: "unannotated",
    };
    if (input.mode === "semantic") {
        record.label_maps = frameNames.map(() => null);
    }

    const dataset = withVideoRecord(
        input.annotation ? await readAnnotationFile(input.annotation) : null,
        record,
    );

    const writer = new ZipWriter();
    for (let i = 0; i < ordered.length; i++) {
        throwIfAborted(input.signal);
        await writer.addBlob(
            `${FRAMES_DIR}/${ordered[i].name}`,
            ordered[i].file,
        );
        input.onProgress?.({
            stage: "frames",
            done: i + 1,
            total: ordered.length,
        });
    }
    if (video) {
        throwIfAborted(input.signal);
        await writer.addBlob(videoEntry as string, video, (loaded, total) =>
            input.onProgress?.({ stage: "video", loaded, total }),
        );
    }
    writer.addText(
        `${ANNOTATIONS_DIR}/${name}.json`,
        JSON.stringify(dataset, null, 2),
    );
    writer.addText(METADATA_ENTRY, JSON.stringify(metadata, null, 2));

    return {
        blob: writer.toBlob(),
        fileName: `${name}${PROJECT_EXTENSION}`,
        name,
        mode: input.mode,
        frameCount: frameNames.length,
        fps,
        source: video ? "video" : "frames",
        frameNames,
    };
}
