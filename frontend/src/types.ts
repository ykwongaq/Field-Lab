export interface RawVideo {
    id: number;
    video_name: string;
    file_names: string[];
    length: number;
    height: number;
    width: number;
    fps: number;
    original_video?: string | null;
    segmentation_mode?: string | null;
    video_file?: string | null;
    /** Frame rate of the source material, when it is known. */
    original_fps?: number | null;
    /** Frame rate the project was created for (intent, not a re-encode). */
    target_fps?: number | null;
    /** Decimation applied when the frames were extracted, if any. */
    frame_step?: number;

    label_maps?: (string | null)[] | null;
    scene_id?: string;
    start_frame?: number;
    end_frame?: number;
    status?: string;
}

export interface RawRle {
    size: [number, number];
    counts: string | number[];
}

export interface ForegroundRun {
    x: number;
    y: number;
    length: number;
}

export interface DecodedMask {
    height: number;
    width: number;
    runs: ForegroundRun[];
}

export interface RawAnnotation {
    id: number;
    video_id: number;
    object_id: number;
    category_id: number;
    noun_phrase: string;

    /**
     * Tracklet layout: one entry per frame (`segmentations[i]` is frame `i`).
     * A `null` entry means the object is not visible on that frame.
     */
    segmentations?: (RawRle | null)[];
    /**
     * Per-frame (COCO-style) layout: the frame the row's single `segmentation`
     * belongs to. Read as a frame index and folded into `segmentations` on open.
     */
    image_id?: number;
    /** Per-frame (COCO-style) layout: the one mask on `image_id`. */
    segmentation?: RawRle | null;
}

export interface RawCategory {
    id: number;
    /**
     * Export name: the common name when one is given, else the deepest
     * specified taxonomic rank, else `null` when nothing is named at all.
     */
    name?: string | null;
    taxon_id?: number;
    kingdom?: string;
    phylum?: string;
    class?: string;
    order?: string;
    family?: string;
    genus?: string;
    species?: string;
    common_name?: string;
}

export interface RawDataset {
    videos: RawVideo[];
    annotations: RawAnnotation[];
    categories?: RawCategory[];
}

export interface Taxonomy {
    taxonId: number | null;
    kingdom: string;
    phylum: string;
    class: string;
    order: string;
    family: string;
    genus: string;
    species: string;
    commonName: string;
}

export type TaxonomyKey = Exclude<keyof Taxonomy, "taxonId">;

/**
 * A project's label: one class, described once and shared by every tracklet
 * assigned to it. `id` is the app's own 0-based sequence — it is the number
 * drawn on the label's colour block and the COCO `category_id` on export.
 * Unlabelled objects use the reserved id -1.
 */
export interface Label {
    id: number;
    /** Display name — the category's common name, else its species. */
    name: string;
    /** Colour of this label's masks and blocks. Frontend-only, never exported. */
    color: string;
    taxonomy: Taxonomy;
}

export type TrackletOrigin = "dataset" | "created";

export interface Tracklet {
    id: number;
    objectId: number;
    categoryId: number;
    /** The label this tracklet is assigned to, or `null` when unlabelled. */
    labelId: number | null;
    label: string;
    taxonomy: Taxonomy;
    color: string;
    segmentations: (RawRle | null)[];
    maskFrames: {
        first: number;
        last: number;
        count: number;
    };
    origin: TrackletOrigin;
}

export interface PromptPoint {
    x: number;
    y: number;
    /** 1 = include this point in the mask, 0 = exclude it. */
    label: 0 | 1;
}

/**
 * Taxonomy corrections made in the workspace, keyed by **label id**.
 *
 * A label is one class, shared by every tracklet assigned to it, so an entry
 * here describes the whole class at once. A label absent from the map keeps the
 * taxonomy the archive gave it, so an empty map means "nothing was relabelled".
 * This is the only annotation state that lives outside the `Clip`, because it is
 * keyed by hand and written on export rather than affecting the masks.
 */
export type TaxonomyOverrides = Record<number, Taxonomy>;
