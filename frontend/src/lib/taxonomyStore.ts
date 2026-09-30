import type { Taxonomy, TaxonomyOverrides, Tracklet } from "../types";

const STORAGE_PREFIX = "vsr.taxonomy.";

/**
 * Taxonomy corrections made in the workspace, persisted to `localStorage` so a
 * half-labelled clip can be resumed later.
 *
 * Only *changes* are stored. A tracklet with no entry keeps the taxonomy the
 * archive gave it, so an untouched clip saves nothing and reopening it is
 * indistinguishable from the first open.
 */
export class TaxonomyStore {
    private readonly clipName: string;
    private overrides: TaxonomyOverrides;

    private constructor(clipName: string, overrides: TaxonomyOverrides) {
        this.clipName = clipName;
        this.overrides = overrides;
    }

    static load(clipName: string): TaxonomyStore {
        let overrides: TaxonomyOverrides = {};
        try {
            const raw = localStorage.getItem(STORAGE_PREFIX + clipName);
            if (raw) overrides = JSON.parse(raw) as TaxonomyOverrides;
        } catch {
            overrides = {};
        }
        return new TaxonomyStore(clipName, overrides);
    }

    /** The taxonomy in force for a tracklet: the edit, else the archive's own. */
    get(tracklet: Tracklet): Taxonomy {
        return this.overrides[tracklet.id] ?? tracklet.taxonomy;
    }

    set(trackletId: number, taxonomy: Taxonomy): void {
        this.overrides[trackletId] = taxonomy;
        this.save();
    }

    remove(trackletId: number): void {
        if (!(trackletId in this.overrides)) return;
        delete this.overrides[trackletId];
        this.save();
    }

    /** Every correction, for export. */
    getRecord(): TaxonomyOverrides {
        return this.overrides;
    }

    private save(): void {
        try {
            localStorage.setItem(
                STORAGE_PREFIX + this.clipName,
                JSON.stringify(this.overrides),
            );
        } catch {
            /* storage unavailable — corrections remain in memory only */
        }
    }
}
