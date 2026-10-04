import type {
    Label,
    RawAnnotation,
    RawCategory,
    RawDataset,
    RawRle,
    RawVideo,
    Taxonomy,
    TaxonomyOverrides,
    Tracklet,
} from "../types";
import type { ZipArchive } from "./zip";
import { ZipWriter } from "./zipWriter";
import {
    colorForIndex,
    nextUnusedLabelColor,
    UNLABELLED_COLOR,
} from "./palette";
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

/**
 * Reserved category id for tracklets that carry no label.
 *
 * Real labels are the app's 0-based sequence (0, 1, 2, …), so -1 can never
 * collide with one. It is written to the annotation JSON as the unlabelled
 * bucket's category; the red block is what makes the gap visible in the app.
 */
export const UNLABELLED_LABEL_ID = -1;

/** A taxonomy with every rank empty — the starting point for a new label. */
export function emptyTaxonomy(): Taxonomy {
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

/** The name a label is shown under: its common name, else its scientific one. */
export function labelNameOf(taxonomy: Taxonomy): string {
    return taxonomy.commonName.trim() || taxonomy.species.trim();
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
    /** The project's label table (instance projects; empty for semantic). */
    labels: Label[];
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
 * Parsed representation of one clip: its frames plus the tracklets to annotate.
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
    /**
     * The project's label table: one row per class, shared by every tracklet
     * assigned to it. Empty for semantic projects, whose classes are tracklets.
     */
    readonly labels: Label[];
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
        this.labels = init.labels;
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

        const maxOf = (values: number[]) =>
            values.length ? Math.max(...values) : 0;

        // Category ids start above every id the archive uses or references.
        const nextCategoryId =
            Math.max(
                maxOf((raw.categories ?? []).map((c) => c.id)),
                maxOf((raw.annotations ?? []).map((a) => a.category_id)),
            ) + 1;

        // Instance projects: collapse the archive's category rows into one label
        // per distinct common name (else species), so a class is described once
        // and shared by every tracklet that uses it. Ids are reused from the
        // archive where possible, which keeps an already-deduplicated project
        // stable across reopen/export cycles. Semantic projects keep their
        // per-class categories untouched and never populate this table.
        const labels: Label[] = [];
        const labelsByKey = new Map<string, Label>();

        const resolveInstanceLabel = (
            category: RawCategory | undefined,
        ): Label | null => {
            const taxonomy = taxonomyFromCategory(
                category,
                category?.species ?? "",
            );
            const name = labelNameOf(taxonomy);
            if (!name) return null;
            const key = `${name}\u0000${taxonomy.species}`.toLowerCase();
            const existing = labelsByKey.get(key);
            if (existing) return existing;
            // Label ids are the app's own 0-based sequence: the number shown on
            // a label is its id. Archive category ids are not reused — export
            // remaps annotations to the label id instead.
            const label: Label = {
                id: labels.length,
                name,
                color: nextUnusedLabelColor(labels.map((item) => item.color)),
                taxonomy,
            };
            labelsByKey.set(key, label);
            labels.push(label);
            return label;
        };

        const tracklets: Tracklet[] = (raw.annotations ?? [])
            .filter((a) => a.video_id === video.id)
            .sort((a, b) => a.id - b.id)
            .map((a, i) => {
                const segmentations: (RawRle | null)[] = (
                    a.segmentations ?? []
                ).map((seg) => (seg && seg.counts ? seg : null));

                const category = taxonomyByCategory.get(a.category_id);

                if (mode === "instance") {
                    const label = resolveInstanceLabel(category);
                    return {
                        id: a.id,
                        objectId: a.object_id,
                        categoryId: label?.id ?? UNLABELLED_LABEL_ID,
                        labelId: label?.id ?? null,
                        label: label?.name ?? NEW_TRACKLET_LABEL,
                        taxonomy: label?.taxonomy ?? emptyTaxonomy(),
                        color: label ? label.color : UNLABELLED_COLOR,
                        segmentations,
                        maskFrames: maskFramesOf(segmentations),
                        origin: "dataset" as const,
                    };
                }

                // Semantic: unchanged — one class per category id.
                const species = category?.species || a.noun_phrase || "";
                const taxonomy = taxonomyFromCategory(category, species);
                return {
                    id: a.id,
                    objectId: a.object_id,
                    categoryId: a.category_id,
                    labelId: a.category_id,
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
                            labelId: classId,
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
            nextCategoryId: Math.max(
                nextCategoryId,
                maxOf(tracklets.map((t) => t.categoryId)) + 1,
            ),
            labels,
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

    /** Look a label up by id; `null` (unlabelled) has no label. */
    labelById(labelId: number | null): Label | null {
        if (labelId === null) return null;
        return this.labels.find((label) => label.id === labelId) ?? null;
    }

    /** The label a tracklet is assigned to, or `null` when it has none. */
    labelFor(tracklet: Tracklet): Label | null {
        return this.labelById(tracklet.labelId);
    }

    /**
     * Add a label to the project without assigning anything to it. Its id is
     * the next position in the list, and it takes the next palette colour unless
     * one is given.
     */
    addLabel(
        name: string,
        taxonomy: Taxonomy = emptyTaxonomy(),
        color: string = nextUnusedLabelColor(
            this.labels.map((label) => label.color),
        ),
    ): { clip: Clip; label: Label } {
        const label: Label = {
            id: this.labels.length,
            name: name.trim() || NEW_TRACKLET_LABEL,
            color,
            taxonomy,
        };
        return {
            clip: this.clone({
                labels: [...this.labels, label],
                editCount: this.editCount + 1,
            }),
            label,
        };
    }

    /**
     * Assign a tracklet to a label, creating the label when no label with that
     * name exists yet. The tracklet adopts the label's name, taxonomy and
     * colour, and leaves the red "unlabelled" look behind.
     */
    assignLabel(
        trackletId: number,
        name: string,
        taxonomy: Taxonomy = emptyTaxonomy(),
    ): { clip: Clip; label: Label } {
        const trimmed = name.trim() || NEW_TRACKLET_LABEL;
        const existing = this.labels.find(
            (item) => item.name.trim().toLowerCase() === trimmed.toLowerCase(),
        );
        if (existing) {
            return {
                clip: this.setLabel(trackletId, existing.id),
                label: existing,
            };
        }
        const { clip, label } = this.addLabel(trimmed, taxonomy);
        return { clip: clip.setLabel(trackletId, label.id), label };
    }

    /**
     * Move a tracklet to a label by id, or to the unlabelled bucket with
     * `null`. The object adopts the label's name, taxonomy and colour.
     */
    setLabel(trackletId: number, labelId: number | null): Clip {
        const index = this.tracklets.findIndex((t) => t.id === trackletId);
        if (index < 0) return this;
        const current = this.tracklets[index];
        if (current.labelId === labelId) return this;

        let next: Tracklet;
        if (labelId === null) {
            next = {
                ...current,
                categoryId: UNLABELLED_LABEL_ID,
                labelId: null,
                label: NEW_TRACKLET_LABEL,
                taxonomy: emptyTaxonomy(),
                color: UNLABELLED_COLOR,
            };
        } else {
            const label = this.labels.find((item) => item.id === labelId);
            if (!label) return this;
            next = {
                ...current,
                categoryId: label.id,
                labelId: label.id,
                label: label.name,
                taxonomy: label.taxonomy,
                color: label.color,
            };
        }
        const tracklets = [...this.tracklets];
        tracklets[index] = next;
        return this.clone({ tracklets, editCount: this.editCount + 1 });
    }

    /**
     * Remove a label. Its objects are not deleted — they fall back to
     * unlabelled (the red block with no number), and the labels after it are
     * renumbered so ids stay contiguous from 0.
     *
     * Because ids are positional, the tracklets are remapped in the same step;
     * callers must renumber any other id-keyed state (the persisted store).
     */
    deleteLabel(labelId: number): Clip {
        const index = this.labels.findIndex((label) => label.id === labelId);
        if (index < 0) return this;
        const labels = this.labels
            .filter((label) => label.id !== labelId)
            .map((label) =>
                label.id > labelId ? { ...label, id: label.id - 1 } : label,
            );
        const tracklets = this.tracklets.map((tracklet) => {
            if (tracklet.labelId === labelId) {
                return {
                    ...tracklet,
                    categoryId: UNLABELLED_LABEL_ID,
                    labelId: null,
                    label: NEW_TRACKLET_LABEL,
                    taxonomy: emptyTaxonomy(),
                    color: UNLABELLED_COLOR,
                };
            }
            if (tracklet.labelId !== null && tracklet.labelId > labelId) {
                const shifted = tracklet.labelId - 1;
                return { ...tracklet, labelId: shifted, categoryId: shifted };
            }
            return tracklet;
        });
        return this.clone({
            labels,
            tracklets,
            editCount: this.editCount + 1,
        });
    }

    /**
     * Update a label's name, colour and/or taxonomy. Every tracklet assigned to
     * it mirrors the change, so one edit describes the whole class.
     *
     * `options.edit` (default `true`) decides whether the change counts as a
     * mask edit; applying stored overrides on open passes `false`, so a project
     * is not reported as edited just because its labels carry saved settings.
     */
    updateLabel(
        labelId: number,
        patch: { name?: string; taxonomy?: Taxonomy; color?: string },
        options: { edit?: boolean } = {},
    ): Clip {
        const index = this.labels.findIndex((label) => label.id === labelId);
        if (index < 0) return this;
        const current = this.labels[index];
        const name = patch.name?.trim() ? patch.name.trim() : current.name;
        const taxonomy = patch.taxonomy ?? current.taxonomy;
        const color = patch.color ?? current.color;
        if (
            name === current.name &&
            taxonomy === current.taxonomy &&
            color === current.color
        )
            return this;
        const labels = [...this.labels];
        labels[index] = { ...current, name, taxonomy, color };
        const tracklets = this.tracklets.map((tracklet) =>
            tracklet.labelId === labelId
                ? { ...tracklet, label: name, taxonomy, color }
                : tracklet,
        );
        return this.clone({
            labels,
            tracklets,
            editCount:
                options.edit === false ? this.editCount : this.editCount + 1,
        });
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
            labels: this.labels,
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

        // Semantic: a class *is* a tracklet, so each new one keeps its own
        // category named by the class string — the original behaviour.
        if (this.mode === "semantic") {
            const tracklet: Tracklet = {
                id: this.nextTrackletId,
                objectId: this.nextObjectId,
                categoryId: this.nextCategoryId,
                labelId: this.nextCategoryId,
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

        // Instance: a blank name means unlabelled. No category is minted for it,
        // and it is drawn in the reserved red so the gap is visible.
        const name = label.trim();
        const named = name !== "" && name !== NEW_TRACKLET_LABEL;
        if (!named) {
            const tracklet: Tracklet = {
                id: this.nextTrackletId,
                objectId: this.nextObjectId,
                categoryId: UNLABELLED_LABEL_ID,
                labelId: null,
                label: NEW_TRACKLET_LABEL,
                taxonomy: emptyTaxonomy(),
                color: UNLABELLED_COLOR,
                segmentations,
                maskFrames: maskFramesOf(segmentations),
                origin: "created",
            };
            const clip = this.clone({
                tracklets: [...this.tracklets, tracklet],
                nextTrackletId: this.nextTrackletId + 1,
                nextObjectId: this.nextObjectId + 1,
                editCount: this.editCount + 1,
                dirtyFrames: this.markDirty(frameIndex),
            });
            return { clip, tracklet };
        }

        // A named label is shared: reuse the one with this name, else create it.
        const existing = this.labels.find(
            (item) => item.name.trim().toLowerCase() === name.toLowerCase(),
        );
        const labelId = existing?.id ?? this.labels.length;
        const labels = existing
            ? this.labels
            : [
                  ...this.labels,
                  {
                      id: labelId,
                      name,
                      color: nextUnusedLabelColor(
                          this.labels.map((label) => label.color),
                      ),
                      taxonomy: emptyTaxonomy(),
                  },
              ];
        const tracklet: Tracklet = {
            id: this.nextTrackletId,
            objectId: this.nextObjectId,
            categoryId: labelId,
            labelId,
            label: existing?.name ?? name,
            taxonomy: existing?.taxonomy ?? emptyTaxonomy(),
            color:
                existing?.color ??
                nextUnusedLabelColor(this.labels.map((label) => label.color)),
            segmentations,
            maskFrames: maskFramesOf(segmentations),
            origin: "created",
        };
        const clip = this.clone({
            tracklets: [...this.tracklets, tracklet],
            labels,
            nextTrackletId: this.nextTrackletId + 1,
            nextObjectId: this.nextObjectId + 1,
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
        taxonomies: TaxonomyOverrides = {},
        onProgress?: LoadProgress,
    ): Promise<Blob> {
        if (this.mode !== "semantic") {
            throw new Error("exportProjectZip is for semantic projects only.");
        }
        const dataset = this.toDataset(taxonomies);
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
     *
     * Instance projects export one category per label (named by its common
     * name), so tracklets that share a label share a category. Semantic
     * projects keep their one-category-per-class behaviour. Taxonomy overrides
     * are still keyed by tracklet id in Phase 0; the first one for a label names
     * that label.
     */
    toDataset(taxonomies: TaxonomyOverrides = {}): RawDataset {
        const video = this.raw.videos[0];
        const others = (this.raw.annotations ?? []).filter(
            (a) => a.video_id !== video.id,
        );
        const semantic = this.mode === "semantic";

        // The taxonomy in force for each label: the stored override (keyed by
        // label id) when there is one, else the label's own.
        const taxonomyByLabel = new Map<number, Taxonomy>();
        for (const label of this.labels) {
            taxonomyByLabel.set(
                label.id,
                taxonomies[label.id] ?? label.taxonomy,
            );
        }

        /** What a tracklet's label is called: its common name, else species. */
        const labelName = (tracklet: Tracklet): string => {
            if (tracklet.labelId === null) return NEW_TRACKLET_LABEL;
            const taxonomy = taxonomyByLabel.get(tracklet.labelId);
            return (taxonomy ? labelNameOf(taxonomy) : "") || tracklet.label;
        };

        const annotations: RawAnnotation[] = this.tracklets.map((tracklet) => ({
            id: tracklet.id,
            video_id: video.id,
            object_id: tracklet.objectId,
            category_id: semantic
                ? tracklet.categoryId
                : (tracklet.labelId ?? UNLABELLED_LABEL_ID),
            // The category is named by its common name, not the archive's noun
            // phrase; this field mirrors it for readers that still expect one.
            noun_phrase: semantic ? tracklet.label : labelName(tracklet),
            // Instance: `null` marks frames where the tracklet is absent (see
            // README). Semantic: masks live in the label maps, not here.
            segmentations: semantic
                ? []
                : tracklet.segmentations.map((seg) => seg ?? null),
        }));

        let categories: RawCategory[];
        if (semantic) {
            // Unchanged: one category per class tracklet.
            categories = [...(this.raw.categories ?? [])];
            const known = new Set(categories.map((c) => c.id));
            for (const tracklet of this.tracklets) {
                if (
                    tracklet.origin !== "created" ||
                    known.has(tracklet.categoryId)
                )
                    continue;
                const taxonomy =
                    taxonomies[tracklet.labelId ?? tracklet.categoryId] ??
                    tracklet.taxonomy;
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
        } else {
            // One category per label. Categories only other videos still use are
            // kept; this video's old (now superseded) rows are dropped, which is
            // what makes the label table the single source of truth.
            const otherIds = new Set(
                (this.raw.annotations ?? [])
                    .filter((a) => a.video_id !== video.id)
                    .map((a) => a.category_id),
            );
            const byId = new Map<number, RawCategory>();
            for (const category of this.raw.categories ?? []) {
                if (otherIds.has(category.id)) byId.set(category.id, category);
            }
            for (const label of this.labels) {
                const taxonomy =
                    taxonomyByLabel.get(label.id) ?? label.taxonomy;
                byId.set(label.id, {
                    id: label.id,
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
                    common_name: labelNameOf(taxonomy) || label.name,
                });
            }
            if (this.tracklets.some((t) => t.labelId === null)) {
                byId.set(UNLABELLED_LABEL_ID, {
                    id: UNLABELLED_LABEL_ID,
                    common_name: NEW_TRACKLET_LABEL,
                });
            }
            categories = [...byId.values()];
        }

        // The frames that exist are the ones the session materialised, not the
        // list the archive recorded when it was packed — a video-only archive
        // records none at all. Writing the clip's own sequence keeps the
        // exported JSON addressable by the masks it carries.
        const exported: RawVideo = {
            ...video,
            file_names: [...this.frameNames],
            length: this.frameNames.length,
            width: this.width,
            height: this.height,
            fps: this.fps,
            end_frame: Math.max(0, this.frameNames.length - 1),
            status: this.editCount > 0 ? "edited" : video.status,
            ...(semantic ? { label_maps: this.currentLabelMaps() } : {}),
        };

        return {
            ...this.raw,
            videos: [exported, ...this.raw.videos.slice(1)],
            annotations: [...others, ...annotations],
            categories,
        };
    }
}
