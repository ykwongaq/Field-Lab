import type { Tracklet } from "../types";
import { Icon } from "../ui";
import styles from "./SelectionTray.module.css";

interface SelectionTrayProps {
    /** The selected tracklets, in selection order. */
    tracklets: Tracklet[];
    primaryId: number | null;
    unit: string;
    canMerge: boolean;
    /** True while a run owns the selection. */
    busy?: boolean;
    /** Why merge is unavailable, when it is. */
    mergeHint?: string;
    onFocus: (id: number) => void;
    onRemove: (id: number) => void;
    onClear: () => void;
    onMerge: () => void;
}

/**
 * The active working set.
 *
 * The object list stays the complete index; this tray is the compact view of
 * what is selected. Keeping both means a reviewer can compare a selected object
 * with an unselected neighbour, and deselecting never moves a row around the
 * sidebar. The primary entry is the one Add/Edit, the timeline and drafts act on;
 * Merge uses the whole set.
 */
export function SelectionTray({
    tracklets,
    primaryId,
    unit,
    canMerge,
    busy = false,
    mergeHint,
    onFocus,
    onRemove,
    onClear,
    onMerge,
}: SelectionTrayProps) {
    if (tracklets.length === 0) return null;

    return (
        <section
            className={styles.panel}
            aria-label={`Selected ${unit}s`}
            aria-busy={busy}
        >
            <header className={styles.head}>
                <span className="sectionLabel">Selection</span>
                <span className={styles.count}>{tracklets.length}</span>
                <button
                    type="button"
                    className={styles.clear}
                    disabled={busy}
                    onClick={onClear}
                    title={
                        busy
                            ? "Selection is locked while a run is live"
                            : "Clear the selection"
                    }
                >
                    Clear
                </button>
            </header>

            <div className={`scrollArea ${styles.list}`}>
                {tracklets.map((tracklet) => {
                    const primary = tracklet.id === primaryId;
                    return (
                        <div
                            key={tracklet.id}
                            className={`${styles.chip} ${
                                primary ? styles.primary : ""
                            }`}
                        >
                            <button
                                type="button"
                                className={styles.chipBody}
                                disabled={busy}
                                onClick={() => onFocus(tracklet.id)}
                                title={
                                    primary
                                        ? "This is the primary selection"
                                        : "Make this the primary selection"
                                }
                            >
                                <span
                                    className={styles.swatch}
                                    style={{ background: tracklet.color }}
                                />
                                <span className={styles.name}>
                                    {tracklet.label}
                                </span>
                                <span className={styles.meta}>
                                    #{tracklet.id} · {tracklet.maskFrames.count}
                                    f
                                </span>
                                {primary && (
                                    <span className={styles.badge}>
                                        primary
                                    </span>
                                )}
                            </button>
                            <button
                                type="button"
                                className={styles.remove}
                                disabled={busy}
                                onClick={() => onRemove(tracklet.id)}
                                aria-label={`Remove ${unit} #${tracklet.id} from the selection`}
                                title={
                                    busy
                                        ? "Selection is locked while a run is live"
                                        : "Remove from selection"
                                }
                            >
                                ×
                            </button>
                        </div>
                    );
                })}
            </div>

            <div className={styles.actions}>
                <button
                    type="button"
                    className="btn btnPrimary btnSmall"
                    disabled={!canMerge || busy}
                    onClick={onMerge}
                    title={
                        busy
                            ? "Stop the run before merging"
                            : (mergeHint ?? "Merge the selected tracklets")
                    }
                >
                    <Icon name="merge" size={14} />
                    {tracklets.length > 1
                        ? `Merge ${tracklets.length}…`
                        : "Merge"}
                </button>
                {mergeHint && <span className={styles.hint}>{mergeHint}</span>}
            </div>
        </section>
    );
}
