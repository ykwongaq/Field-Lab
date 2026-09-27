import type { RawVideo } from "../types";

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