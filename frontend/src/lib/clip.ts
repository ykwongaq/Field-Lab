import type {
    RawAnnotation,
    RawCategory,
    RawDataset,
    RawRle,
    Taxonomy,
    Tracklet,
    TrackletReview,
} from "../types";
import type { ZipArchive } from "./zip";
import { ZipWriter } from "./zipWriter";
import { colorForIndex } from "./palette";
import { ANNOTATION_ENTRY, readProjectMode, type ProjectMode } from "./project";
import { rleIsEmpty, subtractRle, unionRle } from "./rle";
import {
    classMasksToLabelMap,
    decodeLabelMapPng,
    encodeLabelMapPng,
    labelMapEntryFor,
    labelMapToClassMasks,
    MAX_LABEL_MAP_CLASS_ID,
} from "./labelMap";

/** Label given to a tracklet created with the "Add mask" tool. */
export const NEW_TRACKLET_LABEL = "unlabelled object";

function emptyTaxonomy(): Taxonomy {
    return {
        taxonId: null,
        kingdom: "",
        phylum: "",
        class: "",
        order: "",
        family: "",
        genus: "",
        species: "",
        commonName: "",
    };
}

function taxonomyFromCategory(
    category: RawCategory | undefined,
    species: string = category?.species ?? "",
): Taxonomy {
    return {
        taxonId: category?.taxon_id ?? null,
        kingdom: category?.kingdom ?? "",
        phylum: category?.phylum ?? "",
        class: category?.class ?? "",
        order: category?.order ?? "",
        family: category?.family ?? "",
        genus: category?.genus ?? "",
        species,
        commonName: category?.common_name ?? "",
    };
}

function maskFramesOf(
    segmentations: (RawRle | null)[],
): Tracklet["maskFrames"] {
    let first = -1;
    let last = -1;
    let count = 0;
    segmentations.forEach((seg, j) => {
        if (seg && seg.counts) {
            count += 1;
            if (first < 0) first = j;
            last = j;
        }
    });
    return { first, last, count };
}

interface ClipInit {
    name: string;
    width: number;
    height: number;
    fps: number;
    frameNames: string[];
    tracklets: Tracklet[];
    mode: ProjectMode;
    modeAssumed: boolean;
    videoEntry: string | null;
    annotationEntry: string;
    raw: RawDataset;
    nextTrackletId: number;
    nextObjectId: number;
    nextCategoryId: number;
    editCount: number;
    /** Semantic projects: `videos[0].label_maps` as read (null on legacy/instance). */
    labelMaps: (string | null)[] | null;
    /** Frames whose masks changed since the archive was opened. */
    dirtyFrames: ReadonlySet<number>;
    /** `true` when the frames were materialised outside the archive. */
    framesExternal: boolean;
}

/** Progress callback while label maps are decoded on open. */
export type LoadProgress = (done: number, total: number) => void;

/**
 * The frame sequence the backend materialised for a project.
 *
 * It wins over the archive's own `file_names` because it describes the frames
 * that actually exist, measured after extraction, rather than what was asked
 * for when the archive was packed.
 */
export interface FrameOverride {
    frameNames: readonly string[];
    fps: number;
    width: number;
    height: number;
    /** `true` when the frames live outside the archive. */
    external?: boolean;
}

export interface OpenOptions {
    onProgress?: LoadProgress;
    frames?: FrameOverride;
}

/** Legacy archives kept the dataset in a folder rather than at the root. */
const LEGACY_ANNOTATION_PATTERN = /^annotations\/[^/]+\.json$/i;

/**
 * The archive entry holding the dataset: `annotation.json` at the root, or —
 * for archives written before the layout was flattened — the single JSON under
 * `annotations/`.
 */
function findAnnotationEntry(zip: ZipArchive): string {
    if (zip.hasEntry(ANNOTATION_ENTRY)) return ANNOTATION_ENTRY;
    for (const name of zip.getEntries()) {
        if (LEGACY_ANNOTATION_PATTERN.test(name)) return name;
    }
    throw new Error(
        `No annotation JSON found: expected ${ANNOTATION_ENTRY} at the root of ` +
            "the archive (older archives may keep it under annotations/).",
    );
}

/**
 * Parsed representation of one clip: its frames plus the tracklets to review.
 *
 * A `Clip` is immutable: the editing methods (`addTracklet`, `removeMask`,
 * `removeTracklet`) return a new `Clip` that shares the frame data, so React
 * state updates stay cheap and referentially clean. `editCount` says how many
 * edits separate the clip from the archive it was read from; `toDataset()`
 * writes the current tracklets back into annotation-JSON form.
 *
 * Semantic projects store masks as per-frame label maps (PNG, pixel value =
 * category id) rather than per-tracklet RLE. They are expanded into one
 * tracklet per class on open so the rest of the app is mode-agnostic, and
 * flattened back by `exportProjectZip()`. Classes never overlap: painting a
 * class on a frame removes those pixels from every other class.
 */
export class Clip {
    readonly name: string;
    readonly width: number;
    readonly height: number;
    readonly fps: number;
    readonly frameNames: string[];
    readonly tracklets: Tracklet[];
    /** Segmentation mode fixed at project creation; read-only by design. */
    readonly mode: ProjectMode;
    /** `true` when the JSON had no `segmentation_mode` (old archive, instance assumed). */
    readonly modeAssumed: boolean;
    /** Archive entry of the embedded source video, when present. */
    readonly videoEntry: string | null;
    /** Archive entry the annotations were read from, usually `annotation.json`. */
    readonly annotationEntry: string;
    /** Number of mask edits made in the app since the archive was opened. */
    readonly editCount: number;
    /** Frames whose masks changed in the app (drives which label maps are rewritten). */
    readonly dirtyFrames: ReadonlySet<number>;
    /** `true` when the frames came from the backend rather than the archive. */
    readonly framesExternal: boolean;

    private readonly raw: RawDataset;
    private readonly labelMaps: (string | null)[] | null;
    // Ids are handed out monotonically and never reused, so a deleted tracklet's
    // cached masks can never be mistaken for a new one's.
    private readonly nextTrackletId: number;
    private readonly nextObjectId: number;
    private readonly nextCategoryId: number;

    private constructor(init: ClipInit) {
        this.name = init.name;
        this.width = init.width;
        this.height = init.height;
        this.fps = init.fps;
        this.frameNames = init.frameNames;
        this.tracklets = init.tracklets;
        this.mode = init.mode;
        this.modeAssumed = init.modeAssumed;
        this.videoEntry = init.videoEntry;
        this.annotationEntry = init.annotationEntry;
        this.raw = init.raw;
        this.nextTrackletId = init.nextTrackletId;
        this.nextObjectId = init.nextObjectId;
        this.nextCategoryId = init.nextCategoryId;
        this.editCount = init.editCount;
        this.labelMaps = init.labelMaps;
        this.dirtyFrames = init.dirtyFrames;
        this.framesExternal = init.framesExternal;
    }

    static async fromZip(
        zip: ZipArchive,
        options: OpenOptions = {},
    ): Promise<Clip> {
        const { onProgress } = options;
        const annotationEntry = findAnnotationEntry(zip);

        const raw = JSON.parse(
            await zip.readAsText(annotationEntry),
        ) as RawDataset;
        const video = raw.videos?.[0];
        if (!video)
            throw new Error("Annotation JSON contains no video record.");

        // The backend materialises the frames and reports the sequence back, so
        // a project opens the same way whether its archive carried a frame
        // folder or a video. Without that report, the archive's own list is the
        // only option — and it exists only when the archive shipped with frames.
        const override = options.frames;
        const frameNames = override
            ? [...override.frameNames]
            : Array.isArray(video.file_names)
              ? [...video.file_names]
              : [];
        if (frameNames.length === 0) {
            throw new Error(
                video.video_file
                    ? `This project carries a source video (${video.video_file}) but no frames. Open it through the backend, which extracts them first.`
                    : "Annotation JSON contains no frame list (file_names).",
            );
        }

        // The session measured the frames it produced; the JSON only records
        // what was asked for, so the session wins wherever it has an answer.
        const width = override?.width || video.width || 0;
        const height = override?.height || video.height || 0;
        const fps =
            override?.fps ||
            (typeof video.fps === "number" && video.fps > 0 ? video.fps : 25);

        // Mode before anything else: an unsupported mode rejects the archive.
        const { mode, assumed: modeAssumed } = readProjectMode(video);
        const videoEntry =
            typeof video.video_file === "string" &&
            zip.hasEntry(video.video_file)
                ? video.video_file
                : null;

        const taxonomyByCategory = new Map<number, RawCategory>(
            (raw.categories ?? []).map((category) => [category.id, category]),
        );

        const tracklets: Tracklet[] = (raw.annotations ?? [])
            .filter((a) => a.video_id === video.id)
            .sort((a, b) => a.id - b.id)
            .map((a, i) => {
                const segmentations: (RawRle | null)[] = (
                    a.segmentations ?? []
                ).map((seg) => (seg && seg.counts ? seg : null));

                const category = taxonomyByCategory.get(a.category_id);
                const species = category?.species || a.noun_phrase || "";
                const taxonomy = taxonomyFromCategory(category, species);

                return {
                    id: a.id,
                    objectId: a.object_id,
                    categoryId: a.category_id,
                    label: a.noun_phrase ?? species ?? `object ${a.object_id}`,
                    taxonomy,
                    color: colorForIndex(i),
                    segmentations,
                    maskFrames: maskFramesOf(segmentations),
                    origin: "dataset" as const,
                };
            });

        // `video_name` is written by every creation path; the entry name is a
        // fallback for hand-made archives, where the fixed root entry carries
        // no project name of its own.
        const name =
            video.video_name ??
            (annotationEntry === ANNOTATION_ENTRY
                ? "clip"
                : annotationEntry.replace(/^.*\//, "").replace(/\.json$/i, ""));

        const maxOf = (values: number[]) =>
            values.length ? Math.max(...values) : 0;

        // Semantic projects: masks live in per-frame label maps, one tracklet
        // per class. (A semantic archive without `label_maps` is read like an
        // instance one — per-tracklet RLE — and converted on export.)
        const labelMaps =
            mode === "semantic" && Array.isArray(video.label_maps)
                ? frameNames.map((_, i) => video.label_maps?.[i] ?? null)
                : null;
        let nextTrackletId =
            maxOf((raw.annotations ?? []).map((a) => a.id)) + 1;
        let nextObjectId =
            maxOf((raw.annotations ?? []).map((a) => a.object_id)) + 1;
        if (labelMaps) {
            const byCategory = new Map(tracklets.map((t) => [t.categoryId, t]));
            const frameCount = frameNames.length;
            for (const tracklet of tracklets) {
                tracklet.segmentations = new Array(frameCount).fill(null);
            }
            const present = labelMaps
                .map((entry, i) => ({ entry, i }))
                .filter(({ entry }) => entry && zip.hasEntry(entry));
            let done = 0;
            for (const { entry, i } of present) {
                const ids = await decodeLabelMapPng(
                    await zip.readAsBlob(entry as string),
                    width,
                    height,
                );
                for (const [classId, rle] of labelMapToClassMasks(
                    ids,
                    width,
                    height,
                )) {
                    let tracklet = byCategory.get(classId);
                    if (!tracklet) {
                        // A class painted in the maps but missing from `annotations`.
                        const category = taxonomyByCategory.get(classId);
                        tracklet = {
                            id: nextTrackletId++,
                            objectId: nextObjectId++,
                            categoryId: classId,
                            label:
                                category?.species ||
                                category?.common_name ||
                                `class ${classId}`,
                            taxonomy: taxonomyFromCategory(category),
                            color: colorForIndex(tracklets.length),
                            segmentations: new Array(frameCount).fill(null),
                            maskFrames: { first: -1, last: -1, count: 0 },
                            origin: "dataset",
                        };
                        tracklets.push(tracklet);
                        byCategory.set(classId, tracklet);
                    }
                    tracklet.segmentations[i] = rle;
                }
                done += 1;
                onProgress?.(done, present.length);
            }
            for (const tracklet of tracklets) {
                tracklet.maskFrames = maskFramesOf(tracklet.segmentations);
            }
        }

        return new Clip({
            name,
            width,
            height,
            fps,
            frameNames,
            framesExternal: override?.external ?? false,
            tracklets,
            mode,
            modeAssumed,
            videoEntry,
            annotationEntry,
            raw,
            nextTrackletId,
            nextObjectId,
            nextCategoryId:
                Math.max(
                    maxOf((raw.categories ?? []).map((c) => c.id)),
                    maxOf(tracklets.map((t) => t.categoryId)),
                ) + 1,
            editCount: 0,
            labelMaps,
            dirtyFrames: new Set(),
        });
    }

    get frameCount(): number {
        return this.frameNames.length;
    }

    /**
     * Frames the archive does not carry. Always empty when the frames came from
     * the backend, which guarantees the sequence it reported.
     */
    missingFrames(zip: ZipArchive): string[] {
        if (this.framesExternal) return [];
        return this.frameNames.filter(
            (name) => !zip.hasEntry(`frames/${name}`),
        );
    }

    rawMaskAt(tracklet: Tracklet, frameIndex: number): RawRle | null {
        const seg = tracklet.segmentations[frameIndex];
        if (!seg || !seg.counts) return null;
        return seg;
    }

    // ----------------------------------------------------------------- editing

    private clone(patch: Partial<ClipInit>): Clip {
        return new Clip({
            name: this.name,
            width: this.width,
            height: this.height,
            fps: this.fps,
            frameNames: this.frameNames,
            tracklets: this.tracklets,
            mode: this.mode,
            modeAssumed: this.modeAssumed,
            videoEntry: this.videoEntry,
            annotationEntry: this.annotationEntry,
            raw: this.raw,
            nextTrackletId: this.nextTrackletId,
            nextObjectId: this.nextObjectId,
            nextCategoryId: this.nextCategoryId,
            editCount: this.editCount,
            labelMaps: this.labelMaps,
            dirtyFrames: this.dirtyFrames,
            framesExternal: this.framesExternal,
            ...patch,
        });
    }

    private markDirty(...frames: number[]): ReadonlySet<number> {
        const next = new Set(this.dirtyFrames);
        for (const frame of frames) next.add(frame);
        return next;
    }

    /**
     * Create a new tracklet whose only mask is `mask` on `frameIndex`.
     *
     * The tracklet gets fresh ids and its own category (so the reviewer's
     * taxonomy can be written to it on export without touching existing
     * categories). Returns the new clip and the created tracklet.
     */
    addTracklet(
        frameIndex: number,
        mask: RawRle,
        label: string = NEW_TRACKLET_LABEL,
    ): { clip: Clip; tracklet: Tracklet } {
        if (frameIndex < 0 || frameIndex >= this.frameCount) {
            throw new Error(`Frame index out of range: ${frameIndex}`);
        }
        const segmentations: (RawRle | null)[] = new Array(
            this.frameCount,
        ).fill(null);
        segmentations[frameIndex] = mask;
        const tracklet: Tracklet = {
            id: this.nextTrackletId,
            objectId: this.nextObjectId,
            categoryId: this.nextCategoryId,
            label,
            taxonomy: emptyTaxonomy(),
            color: colorForIndex(this.tracklets.length),
            segmentations,
            maskFrames: maskFramesOf(segmentations),
            origin: "created",
        };
        const clip = this.clone({
            tracklets: [...this.tracklets, tracklet],
            nextTrackletId: this.nextTrackletId + 1,
            nextObjectId: this.nextObjectId + 1,
            nextCategoryId: this.nextCategoryId + 1,
            editCount: this.editCount + 1,
            dirtyFrames: this.markDirty(frameIndex),
        });
        return { clip, tracklet };
    }

    /** Set (or replace) a tracklet's mask on one frame. */
    setMask(trackletId: number, frameIndex: number, mask: RawRle): Clip {
        return this.updateSegmentations(
            trackletId,
            (segmentations) => {
                segmentations[frameIndex] = mask;
            },
            frameIndex,
        );
    }

    /**
     * Remove a tracklet's mask on one frame. A tracklet left without any mask
     * is removed entirely.
     */
    removeMask(trackletId: number, frameIndex: number): Clip {
        const tracklet = this.tracklets.find((t) => t.id === trackletId);
        if (!tracklet || !this.rawMaskAt(tracklet, frameIndex)) return this;
        if (tracklet.maskFrames.count <= 1)
            return this.removeTracklet(trackletId);
        return this.updateSegmentations(
            trackletId,
            (segmentations) => {
                segmentations[frameIndex] = null;
            },
            frameIndex,
        );
    }

    /** Remove a whole tracklet. */
    removeTracklet(trackletId: number): Clip {
        const tracklet = this.tracklets.find((t) => t.id === trackletId);
        if (!tracklet) return this;
        const touched: number[] = [];
        tracklet.segmentations.forEach((seg, i) => {
            if (seg && seg.counts) touched.push(i);
        });
        return this.clone({
            tracklets: this.tracklets.filter((t) => t.id !== trackletId),
            editCount: this.editCount + 1,
            dirtyFrames: this.markDirty(...touched),
        });
    }

    private updateSegmentations(
        trackletId: number,
        mutate: (segmentations: (RawRle | null)[]) => void,
        frameIndex: number,
    ): Clip {
        const index = this.tracklets.findIndex((t) => t.id === trackletId);
        if (index < 0) return this;
        const current = this.tracklets[index];
        const segmentations = [...current.segmentations];
        while (segmentations.length < this.frameCount) segmentations.push(null);
        mutate(segmentations);
        const next: Tracklet = {
            ...current,
            segmentations,
            maskFrames: maskFramesOf(segmentations),
        };
        const tracklets = [...this.tracklets];
        tracklets[index] = next;
        return this.clone({
            tracklets,
            editCount: this.editCount + 1,
            dirtyFrames: this.markDirty(frameIndex),
        });
    }

    // --------------------------------------------------- semantic (classes)

    /** Case-insensitive lookup of a class/tracklet by its label. */
    findTrackletByLabel(label: string): Tracklet | null {
        const needle = label.trim().toLowerCase();
        if (!needle) return null;
        return (
            this.tracklets.find(
                (t) => t.label.trim().toLowerCase() === needle,
            ) ?? null
        );
    }

    /** Semantic projects: can one more class still fit in an 8-bit label map? */
    get canAddClass(): boolean {
        return (
            this.mode !== "semantic" ||
            this.nextCategoryId <= MAX_LABEL_MAP_CLASS_ID
        );
    }

    /**
     * Semantic "Add mask": union `mask` into class `trackletId` on `frameIndex`
     * and take those pixels away from every other class on that frame, so a
     * pixel always has exactly one class (a label map cannot say otherwise).
     * Classes left without any mask are removed.
     */
    paintClass(frameIndex: number, trackletId: number, mask: RawRle): Clip {
        const target = this.tracklets.find((t) => t.id === trackletId);
        if (!target) throw new Error(`Unknown ${this.mode} id ${trackletId}`);
        let clip: Clip = this.setMask(
            trackletId,
            frameIndex,
            unionRle(this.rawMaskAt(target, frameIndex), mask),
        );
        clip = clip.carveOthers(frameIndex, trackletId, mask);
        // Count the whole paint as a single edit.
        return clip.clone({ editCount: this.editCount + 1 });
    }

    /**
     * Semantic "Add mask" for a class that does not exist yet: create the
     * tracklet with `mask` on `frameIndex`, then carve it out of the others.
     */
    addClass(
        frameIndex: number,
        mask: RawRle,
        label: string,
    ): { clip: Clip; tracklet: Tracklet } {
        if (!this.canAddClass) {
            throw new Error(
                `A semantic project can hold at most ${MAX_LABEL_MAP_CLASS_ID} classes (8-bit label maps).`,
            );
        }
        const { clip, tracklet } = this.addTracklet(frameIndex, mask, label);
        const carved = clip
            .carveOthers(frameIndex, tracklet.id, mask)
            .clone({ editCount: this.editCount + 1 });
        return {
            clip: carved,
            tracklet:
                carved.tracklets.find((t) => t.id === tracklet.id) ?? tracklet,
        };
    }

    replaceMask(
        trackletId: number,
        frameIndex: number,
        mask: RawRle | null,
    ): Clip {
        const target = this.tracklets.find((t) => t.id === trackletId);
        if (!target) throw new Error(`Unknown ${this.mode} id ${trackletId}`);
        if (!mask || rleIsEmpty(mask)) {
            return this.removeMask(trackletId, frameIndex);
        }
        let clip: Clip = this.setMask(trackletId, frameIndex, mask);
        if (this.mode === "semantic") {
            clip = clip.carveOthers(frameIndex, trackletId, mask);
        }
        // Count the whole edit as a single change.
        return clip.clone({ editCount: this.editCount + 1 });
    }

    private carveOthers(
        frameIndex: number,
        keepId: number,
        mask: RawRle,
    ): Clip {
        let clip: Clip = this;
        for (const other of this.tracklets) {
            if (other.id === keepId) continue;
            const existing = this.rawMaskAt(other, frameIndex);
            if (!existing) continue;
            const remaining = subtractRle(existing, mask);
            if (remaining === existing) continue;
            clip = remaining
                ? clip.setMask(other.id, frameIndex, remaining)
                : clip.removeMask(other.id, frameIndex);
        }
        return clip;
    }

    /** Frames on which at least one tracklet has a mask. */
    private framesWithMasks(): Set<number> {
        const frames = new Set<number>();
        for (const tracklet of this.tracklets) {
            tracklet.segmentations.forEach((seg, i) => {
                if (seg && seg.counts && !rleIsEmpty(seg)) frames.add(i);
            });
        }
        return frames;
    }

    /** Semantic projects: `videos[0].label_maps` for the current tracklets. */
    private currentLabelMaps(): (string | null)[] {
        const withMasks = this.framesWithMasks();
        return this.frameNames.map((name, i) =>
            withMasks.has(i)
                ? (this.labelMaps?.[i] ?? labelMapEntryFor(name))
                : null,
        );
    }

    /** Row-major label map of one frame (category id per pixel). */
    labelMapOf(frameIndex: number): Uint8Array {
        const masks: { id: number; rle: RawRle }[] = [];
        for (const tracklet of this.tracklets) {
            const rle = this.rawMaskAt(tracklet, frameIndex);
            if (rle) masks.push({ id: tracklet.categoryId, rle });
        }
        return classMasksToLabelMap(masks, this.width, this.height);
    }

    /**
     * Semantic projects: write the whole project archive back out — frames and
     * video copied from `zip` untouched, the annotation JSON replaced, and a
     * label-map PNG for every frame whose masks changed (or that never had one).
     */
    async exportProjectZip(
        zip: ZipArchive,
        reviews: Record<number, TrackletReview> = {},
        onProgress?: LoadProgress,
    ): Promise<Blob> {
        if (this.mode !== "semantic") {
            throw new Error("exportProjectZip is for semantic projects only.");
        }
        const dataset = this.toDataset(reviews);
        const labelMaps = dataset.videos[0].label_maps ?? [];
        const writer = new ZipWriter();
        writer.addText(this.annotationEntry, JSON.stringify(dataset, null, 2));

        // Label maps: reuse the stored PNG when the frame did not change.
        const toWrite: number[] = [];
        for (let i = 0; i < labelMaps.length; i++) {
            const entry = labelMaps[i];
            if (!entry) continue;
            const unchanged =
                !this.dirtyFrames.has(i) &&
                this.labelMaps?.[i] === entry &&
                zip.hasEntry(entry);
            if (unchanged) writer.addRaw(await zip.rawEntry(entry));
            else toWrite.push(i);
        }
        let done = 0;
        for (const i of toWrite) {
            const entry = labelMaps[i] as string;
            const png = await encodeLabelMapPng(
                this.labelMapOf(i),
                this.width,
                this.height,
            );
            await writer.addBlob(entry, png);
            done += 1;
            onProgress?.(done, toWrite.length);
        }

        // Everything else (frames/, video/, extras) is copied as-is. Old label
        // maps of frames that no longer have masks are dropped on purpose.
        for (const name of zip.getEntries()) {
            if (writer.has(name)) continue;
            if (name === this.annotationEntry) continue;
            if (/^masks\/[^/]+\.png$/i.test(name)) continue;
            if (name.endsWith("/")) continue; // directory placeholders
            writer.addRaw(await zip.rawEntry(name));
        }
        return writer.toBlob();
    }

    // ------------------------------------------------------------------ export

    /**
     * Write the current tracklets back into the annotation-JSON layout.
     *
     * Annotations for other videos in the file (if any) are kept untouched.
     * Categories created in the app (one per added tracklet) are filled from
     * the reviewer's final taxonomy when `reviews` is given.
     */
    toDataset(reviews: Record<number, TrackletReview> = {}): RawDataset {
        const video = this.raw.videos[0];
        const others = (this.raw.annotations ?? []).filter(
            (a) => a.video_id !== video.id,
        );
        const semantic = this.mode === "semantic";
        const annotations: RawAnnotation[] = this.tracklets.map((tracklet) => ({
            id: tracklet.id,
            video_id: video.id,
            object_id: tracklet.objectId,
            category_id: tracklet.categoryId,
            noun_phrase: tracklet.label,
            // Instance: `null` marks frames where the tracklet is absent (see
            // README). Semantic: masks live in the label maps, not here.
            segmentations: semantic
                ? []
                : tracklet.segmentations.map((seg) => seg ?? null),
        }));

        const categories: RawCategory[] = [...(this.raw.categories ?? [])];
        const known = new Set(categories.map((c) => c.id));
        for (const tracklet of this.tracklets) {
            if (tracklet.origin !== "created" || known.has(tracklet.categoryId))
                continue;
            const taxonomy =
                reviews[tracklet.id]?.taxonomy ?? tracklet.taxonomy;
            categories.push({
                id: tracklet.categoryId,
                ...(taxonomy.taxonId !== null
                    ? { taxon_id: taxonomy.taxonId }
                    : {}),
                kingdom: taxonomy.kingdom,
                phylum: taxonomy.phylum,
                class: taxonomy.class,
                order: taxonomy.order,
                family: taxonomy.family,
                genus: taxonomy.genus,
                species: taxonomy.species,
                common_name: taxonomy.commonName,
            });
            known.add(tracklet.categoryId);
        }

        return {
            ...this.raw,
            videos: [
                {
                    ...video,
                    status: this.editCount > 0 ? "edited" : video.status,
                    ...(semantic
                        ? { label_maps: this.currentLabelMaps() }
                        : {}),
                },
                ...this.raw.videos.slice(1),
            ],
            annotations: [...others, ...annotations],
            categories,
        };
    }
}
