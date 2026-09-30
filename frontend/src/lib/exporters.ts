/**
 * The artifacts a reviewed project can hand back.
 *
 * Four things can leave the workspace, one file each:
 *
 * * the **source video** the archive embedded — copied out verbatim, never
 *   re-encoded, so it keeps the provenance of the clip as it arrived;
 * * the **original frames** the archive carried (the folder it was packed
 *   with);
 * * the **sampled frames** the backend materialised and the reviewer actually
 *   annotated — the same images when the archive carried a frame folder, the
 *   frames decoded from the video when it carried one instead;
 * * the **annotation JSON**, i.e. the dataset as it stands after review, in the
 *   COCO video-segmentation layout the project uses everywhere else — or, for a
 *   semantic project, the whole archive it is part of.
 *
 * Every byte already lives in the browser — the archive the project was opened
 * from, the session frame source, and the `Clip` — so an export never needs a
 * round trip to the backend.
 *
 * Each builder returns the finished file rather than saving it, so the caller
 * owns the download and the packing logic can be exercised on its own.
 */

import type { TrackletReview } from "../types";
import type { Clip } from "./clip";
import type { FrameSource } from "./frames";
import { naturalCompare } from "./projectWriter";
import type { ZipArchive } from "./zip";
import { ZipWriter } from "./zipWriter";

/** Where the archive keeps the frames it was packed with. */
const FRAMES_DIR = "frames";

/** Called with `(done, total)` while a many-file export is being written. */
export type ExportProgress = (done: number, total: number) => void;

/**
 * One artifact ready to be saved.
 *
 * Building and saving are kept apart on purpose: the builders are pure enough
 * to exercise outside a browser, and the caller decides when the file actually
 * leaves the page.
 */
export interface ExportedFile {
    fileName: string;
    blob: Blob;
}

/** The last path segment of an archive entry. */
function entryName(entry: string): string {
    return entry.replace(/\\/g, "/").split("/").pop() ?? entry;
}

/** Keep a frame's own base name, so an extracted folder mimics the source. */
function frameName(name: string, index: number): string {
    const base = entryName(name);
    return base || `frame_${String(index).padStart(8, "0")}.jpg`;
}

function extensionOf(entry: string): string {
    const match = /\.[^./\\]+$/.exec(entryName(entry));
    return match ? match[0] : ".mp4";
}

/** Frames of one clip, in playback order. */
function orderFrameNames(names: readonly string[]): string[] {
    return [...names].sort(naturalCompare);
}

// ------------------------------------------------------------------ original

/** The archive entry of the embedded source video, or `null` when there is none. */
export function sourceVideoEntry(clip: Clip, zip: ZipArchive): string | null {
    const entry = clip.videoEntry;
    return entry && zip.hasEntry(entry) ? entry : null;
}

/** The archive entries of the frames the archive itself carried. */
export function archiveFrameEntries(zip: ZipArchive): string[] {
    return zip
        .getEntries()
        .filter(
            (name) => name.startsWith(`${FRAMES_DIR}/`) && !name.endsWith("/"),
        );
}

/**
 * Build the source video the archive embedded.
 *
 * The bytes are copied out of the archive untouched, so a multi-gigabyte video
 * is downloaded without being decoded or re-encoded.
 */
export async function exportSourceVideo(
    clip: Clip,
    zip: ZipArchive,
): Promise<ExportedFile> {
    const entry = sourceVideoEntry(clip, zip);
    if (!entry) {
        throw new Error("This project carries no source video.");
    }
    const blob = await zip.readAsBlob(entry);
    return { fileName: `${clip.name}${extensionOf(entry)}`, blob };
}

/**
 * Build the frame folder the archive was packed with.
 *
 * Entries are copied between archives with `addRaw`, so a DEFLATE-compressed
 * frame is never inflated just to be stored again — only its name moves to the
 * root of the download, which is what makes the result a plain frame folder.
 */
export async function exportOriginalFrames(
    clip: Clip,
    zip: ZipArchive,
    onProgress?: ExportProgress,
): Promise<ExportedFile> {
    const entries = archiveFrameEntries(zip);
    if (entries.length === 0) {
        throw new Error("This project carries no frames folder.");
    }
    const ordered = orderFrameNames(entries.map(entryName));
    const byName = new Map(entries.map((entry) => [entryName(entry), entry]));

    const writer = new ZipWriter();
    let done = 0;
    for (const name of ordered) {
        const entry = byName.get(name) as string;
        writer.addRaw({ ...(await zip.rawEntry(entry)), name });
        onProgress?.(++done, ordered.length);
    }
    return { fileName: `${clip.name}_frames.zip`, blob: writer.toBlob() };
}

/**
 * Build the frames the reviewer annotated.
 *
 * These come from the session, not the archive, so a video-only project exports
 * the frames that were decoded from it and every mask lines up with the images
 * that were on screen.
 */
export async function exportSampledFrames(
    clip: Clip,
    frames: FrameSource,
    onProgress?: ExportProgress,
): Promise<ExportedFile> {
    if (frames.count === 0) {
        throw new Error("This project has no frames to export.");
    }
    const writer = new ZipWriter();
    for (let index = 0; index < frames.count; index++) {
        const blob = await frames.frame(index);
        await writer.addBlob(frameName(frames.names[index] ?? "", index), blob);
        onProgress?.(index + 1, frames.count);
    }
    return {
        fileName: `${clip.name}_sampled_frames.zip`,
        blob: writer.toBlob(),
    };
}

// ---------------------------------------------------------------- annotation

/**
 * Build the annotation JSON — the dataset as it stands after review.
 *
 * Instance projects only: a semantic project's masks are per-frame label-map
 * PNGs that the JSON points at by name, so it is handed back as an archive
 * instead (see `exportProjectArchive`).
 */
export function exportAnnotation(
    clip: Clip,
    reviews: Record<number, TrackletReview>,
): ExportedFile {
    if (clip.mode === "semantic") {
        throw new Error(
            "A semantic project exports its label maps as a project archive.",
        );
    }
    const json = JSON.stringify(clip.toDataset(reviews), null, 2);
    return {
        fileName: `${clip.name}_annotation.json`,
        blob: new Blob([json], { type: "application/json" }),
    };
}

/**
 * Build a complete project archive from the current review.
 *
 * This is what a semantic project exports: its masks are label-map PNGs that
 * `annotation.json` refers to by entry name, so only a full archive (frames,
 * label maps, annotation and metadata) is self-consistent — and reopening it
 * resumes from the saved state.
 */
export async function exportProjectArchive(
    clip: Clip,
    zip: ZipArchive,
    reviews: Record<number, TrackletReview>,
    onProgress?: ExportProgress,
): Promise<ExportedFile> {
    const blob = await clip.exportProjectZip(zip, reviews, onProgress);
    return { fileName: `${clip.name}.project`, blob };
}
