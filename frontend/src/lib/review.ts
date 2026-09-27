import type { Clip } from "./clip";
import type { Tracklet, TrackletReview } from "../types";
import type { ProjectMode } from "./project";

const STORAGE_PREFIX = "vsr.review.";

export interface ReviewCounts {
	total: number;
	verified: number;
	partial: number;
	pending: number;
}

export interface ExportRow {
	clip: string;
	project_mode: ProjectMode;
	tracklet_id: number;
	object_id: number;
	category_id: number;
	original_label: string;
	original_taxon_id: string;
	original_kingdom: string;
	original_phylum: string;
	original_class: string;
	original_order: string;
	original_family: string;
	original_genus: string;
	original_species: string;
	original_common_name: string;
	label_confirmed: string;
	final_kingdom: string;
	final_phylum: string;
	final_class: string;
	final_order: string;
	final_family: string;
	final_genus: string;
	final_species: string;
	final_common_name: string;
	mask_verdict: string;
	comment: string;
}

export interface ExportPayload {
	clip: string;
	mode: ProjectMode;
	exportedAt: string;
	fps: number;
	width: number;
	height: number;
	reviews: ExportRow[];
	csv: string;
}

const EMPTY_REVIEW: TrackletReview = {
	labelConfirmed: false,
	taxonomy: null,
	maskVerdict: null,
	comment: "",
};

/**
 * Per-clip review state, persisted to `localStorage` so a partially reviewed
 * clip can be resumed later.
 */
export class ReviewStore {
	private readonly clipName: string;
	private reviews: Record<number, TrackletReview>;

	private constructor(
		clipName: string,
		reviews: Record<number, TrackletReview>,
	) {
		this.clipName = clipName;
		this.reviews = reviews;
	}

	static load(clipName: string): ReviewStore {
		let reviews: Record<number, TrackletReview> = {};
		try {
			const raw = localStorage.getItem(STORAGE_PREFIX + clipName);
			if (raw) reviews = JSON.parse(raw) as Record<number, TrackletReview>;
		} catch {
			reviews = {};
		}
		return new ReviewStore(clipName, reviews);
	}

	get(trackletId: number): TrackletReview {
		return this.reviews[trackletId] ?? EMPTY_REVIEW;
	}

	getRecord(): Record<number, TrackletReview> {
		return this.reviews;
	}

	update(trackletId: number, patch: Partial<TrackletReview>): void {
		this.reviews[trackletId] = { ...this.get(trackletId), ...patch };
		this.save();
	}

	remove(trackletId: number): void {
		if (!(trackletId in this.reviews)) return;
		delete this.reviews[trackletId];
		this.save();
	}

	counts(tracklets: Tracklet[]): ReviewCounts {
		let verified = 0;
		let partial = 0;
		let pending = 0;
		for (const tracklet of tracklets) {
			const review = this.get(tracklet.id);
			const hasLabel = review.labelConfirmed;
			const hasMask = review.maskVerdict !== null;
			if (hasLabel && hasMask) verified += 1;
			else if (hasLabel || hasMask) partial += 1;
			else pending += 1;
		}
		return { total: tracklets.length, verified, partial, pending };
	}

	private save(): void {
		try {
			localStorage.setItem(
				STORAGE_PREFIX + this.clipName,
				JSON.stringify(this.reviews),
			);
		} catch {
			/* storage unavailable — reviews remain in memory only */
		}
	}

	static buildExport(
		clip: Clip,
		record: Record<number, TrackletReview>,
	): ExportPayload {
		const header = [
			"clip",
			"project_mode",
			"tracklet_id",
			"object_id",
			"category_id",
			"original_label",
			"original_taxon_id",
			"original_kingdom",
			"original_phylum",
			"original_class",
			"original_order",
			"original_family",
			"original_genus",
			"original_species",
			"original_common_name",
			"label_confirmed",
			"final_kingdom",
			"final_phylum",
			"final_class",
			"final_order",
			"final_family",
			"final_genus",
			"final_species",
			"final_common_name",
			"mask_verdict",
			"comment",
		] as const;

		const rows: ExportRow[] = clip.tracklets.map((tracklet) => {
			const review = record[tracklet.id] ?? EMPTY_REVIEW;
			const original = tracklet.taxonomy;
			const final = review.taxonomy ?? tracklet.taxonomy;
			return {
				clip: clip.name,
				project_mode: clip.mode,
				tracklet_id: tracklet.id,
				object_id: tracklet.objectId,
				category_id: tracklet.categoryId,
				original_label: tracklet.label,
				original_taxon_id: toCell(original.taxonId),
				original_kingdom: original.kingdom,
				original_phylum: original.phylum,
				original_class: original.class,
				original_order: original.order,
				original_family: original.family,
				original_genus: original.genus,
				original_species: original.species,
				original_common_name: original.commonName,
				label_confirmed: String(review.labelConfirmed),
				final_kingdom: final.kingdom,
				final_phylum: final.phylum,
				final_class: final.class,
				final_order: final.order,
				final_family: final.family,
				final_genus: final.genus,
				final_species: final.species,
				final_common_name: final.commonName,
				mask_verdict: review.maskVerdict ?? "",
				comment: review.comment,
			};
		});

		const csv = [header.join(",")]
			.concat(
				rows.map((row) =>
					header.map((key) => escapeCsv(String(row[key]))).join(","),
				),
			)
			.join("\n");

		return {
			clip: clip.name,
			mode: clip.mode,
			exportedAt: new Date().toISOString(),
			fps: clip.fps,
			width: clip.width,
			height: clip.height,
			reviews: rows,
			csv,
		};
	}
}

function escapeCsv(value: string): string {
	if (/[",\n]/.test(value)) return `"${value.replace(/"/g, '""')}"`;
	return value;
}

function toCell(value: string | number | null | undefined): string {
	return value === null || value === undefined ? "" : String(value);
}
