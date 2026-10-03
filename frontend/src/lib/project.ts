import type { RawVideo } from "../types";

/** Archive entry holding the clip's annotation dataset (COCO video format). */
export const ANNOTATION_ENTRY = "annotation.json";

export type ProjectMode = "instance" | "semantic";
export const PROJECT_MODES: readonly ProjectMode[] = ["instance", "semantic"];

export function isProjectMode(value: unknown): value is ProjectMode {
    return (
        typeof value === "string" &&
        (PROJECT_MODES as readonly string[]).includes(value)
    );
}

export interface ModeVocabulary {
    title: string;
    description: string;
    unit: string;
    units: string;
    hasObjectIdentity: boolean;
}

export const MODE_VOCABULARY: Record<ProjectMode, ModeVocabulary> = {
    instance: {
        title: "Instance",
        description:
            "Every tracked object gets its own tracklet with a per-frame mask; several tracklets may share a category.",
        unit: "tracklet",
        units: "tracklets",
        hasObjectIdentity: true,
    },
    semantic: {
        title: "Semantic",
        description:
            "One mask sequence per category. Pixels are labelled by class only; individual objects are not distinguished.",
        unit: "class",
        units: "classes",
        hasObjectIdentity: false,
    },
};

export const DEFAULT_PROJECT_MODE: ProjectMode = "instance";

/**
 * Modes that cannot be chosen when creating a project yet.
 *
 * Opening an existing project of a locked mode still works — this only gates
 * creation. Semantic mode is locked while the label-list rework lands.
 */
export const LOCKED_MODES: readonly ProjectMode[] = ["semantic"];

/**
 * Read the mode from a video record.
 */
export function readProjectMode(video: RawVideo): {
    mode: ProjectMode;
    assumed: boolean;
} {
    const raw = video.segmentation_mode;
    if (raw === undefined || raw === null) {
        return { mode: DEFAULT_PROJECT_MODE, assumed: true };
    }
    if (!isProjectMode(raw)) {
        throw new Error(
            `Annotation JSON declares an unsupported segmentation_mode (${JSON.stringify(
                raw,
            )}). Expected "instance" or "semantic".`,
        );
    }
    return { mode: raw, assumed: false };
}
