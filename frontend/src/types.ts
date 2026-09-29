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

    segmentations?: (RawRle | null)[];
}

export interface RawCategory {
    id: number;
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

export type TrackletOrigin = "dataset" | "created";

export interface Tracklet {
    id: number;
    objectId: number;
    categoryId: number;
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

export type MaskVerdict = "good" | "bad" | "unsure";

export interface TrackletReview {
    labelConfirmed: boolean;
    taxonomy: Taxonomy | null;
    maskVerdict: MaskVerdict | null;
    comment: string;
}
