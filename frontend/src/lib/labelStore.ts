import type { Taxonomy, TaxonomyOverrides } from "../types";
import { labelNameOf, type Clip } from "./clip";

const STORAGE_PREFIX = "vsr.labels.";
/** Colour overrides for labels, kept apart from the taxonomy record. */
const COLOR_PREFIX = "vsr.labelColors.";
/** The pre-label format, which keyed overrides by tracklet id. */
const LEGACY_PREFIX = "vsr.taxonomy.";

/**
 * Workspace edits to labels, persisted to `localStorage` so a half-labelled
 * clip can be resumed later.
 *
 * Two records, both keyed by **label id**: the taxonomy in force (which can
 * differ from the archive's own) and the label's colour. Only *changes* are
 * stored — an untouched clip saves nothing and reopening it is
 * indistinguishable from the first open.
 */
export class LabelStore {
    private readonly clipName: string;
    private overrides: TaxonomyOverrides;
    private colors: Record<number, string>;

    private constructor(
        clipName: string,
        overrides: TaxonomyOverrides,
        colors: Record<number, string>,
    ) {
        this.clipName = clipName;
        this.overrides = overrides;
        this.colors = colors;
    }

    static load(clipName: string, clip: Clip): LabelStore {
        const colors = readColors(COLOR_PREFIX + clipName);
        const stored = read(STORAGE_PREFIX + clipName);
        if (stored) return new LabelStore(clipName, stored, colors);

        // Migrate the pre-label format, whose overrides were keyed by tracklet
        // id. A label shared by several tracklets takes the first of their
        // entries; the rest described the same class and are folded away.
        const legacy = read(LEGACY_PREFIX + clipName);
        if (legacy) {
            const migrated: TaxonomyOverrides = {};
            for (const tracklet of clip.tracklets) {
                if (tracklet.labelId === null) continue;
                const taxonomy = legacy[tracklet.id];
                if (taxonomy && migrated[tracklet.labelId] === undefined) {
                    migrated[tracklet.labelId] = taxonomy;
                }
            }
            const store = new LabelStore(clipName, migrated, colors);
            store.save();
            remove(LEGACY_PREFIX + clipName);
            return store;
        }

        return new LabelStore(clipName, {}, colors);
    }

    /** The taxonomy in force for a label: the edit, else the archive's own. */
    get(label: { id: number; taxonomy: Taxonomy }): Taxonomy {
        return this.overrides[label.id] ?? label.taxonomy;
    }

    /** The colour in force for a label: the edit, else its assigned colour. */
    colorOf(label: { id: number; color: string }): string {
        return this.colors[label.id] ?? label.color;
    }

    set(labelId: number, taxonomy: Taxonomy): void {
        this.overrides[labelId] = taxonomy;
        this.save();
    }

    setColor(labelId: number, color: string): void {
        this.colors[labelId] = color;
        this.saveColors();
    }

    remove(labelId: number): void {
        if (labelId in this.overrides) delete this.overrides[labelId];
        if (labelId in this.colors) delete this.colors[labelId];
        this.save();
        this.saveColors();
    }

    /**
     * Forget a label and shift the ids after it down by one, matching
     * `Clip.deleteLabel`'s renumbering of the label list. Ids are positional, so
     * a saved edit must follow the label it belongs to.
     */
    deleteLabel(labelId: number): void {
        const shift = <T>(record: Record<number, T>): Record<number, T> => {
            const next: Record<number, T> = {};
            for (const [key, value] of Object.entries(record)) {
                const id = Number(key);
                if (id === labelId) continue;
                next[id > labelId ? id - 1 : id] = value;
            }
            return next;
        };
        this.overrides = shift(this.overrides);
        this.colors = shift(this.colors);
        this.save();
        this.saveColors();
    }

    /** Every correction, for export (keyed by label id). */
    getRecord(): TaxonomyOverrides {
        return this.overrides;
    }

    /**
     * Fold the stored edits into a freshly opened clip's labels.
     *
     * The clip is the source of truth for display and export, so a resumed clip
     * should already show its saved taxonomy and colours. Applying them does not
     * count as an edit — the project is not "dirty" simply because its labels
     * were named.
     */
    applyTo(clip: Clip): Clip {
        let next = clip;
        for (const label of clip.labels) {
            const override = this.overrides[label.id];
            const color = this.colors[label.id];
            if (!override && !color) continue;
            next = next.updateLabel(
                label.id,
                {
                    taxonomy: override ?? label.taxonomy,
                    name: override
                        ? labelNameOf(override) || label.name
                        : label.name,
                    color: color ?? label.color,
                },
                { edit: false },
            );
        }
        return next;
    }

    private save(): void {
        write(STORAGE_PREFIX + this.clipName, this.overrides);
    }

    private saveColors(): void {
        write(COLOR_PREFIX + this.clipName, this.colors);
    }
}

function read(key: string): TaxonomyOverrides | null {
    try {
        const raw = localStorage.getItem(key);
        return raw ? (JSON.parse(raw) as TaxonomyOverrides) : null;
    } catch {
        return null;
    }
}

function readColors(key: string): Record<number, string> {
    try {
        const raw = localStorage.getItem(key);
        return raw ? (JSON.parse(raw) as Record<number, string>) : {};
    } catch {
        return {};
    }
}

function write(key: string, value: unknown): void {
    try {
        localStorage.setItem(key, JSON.stringify(value));
    } catch {
        /* storage unavailable — edits remain in memory only */
    }
}

function remove(key: string): void {
    try {
        localStorage.removeItem(key);
    } catch {
        /* storage unavailable */
    }
}
