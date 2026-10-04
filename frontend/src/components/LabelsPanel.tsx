import { useMemo } from "react";
import type { Clip } from "../lib/clip";
import { EmptyState, Icon } from "../ui";
import styles from "./LabelsPanel.module.css";

interface LabelsPanelProps {
    clip: Clip;
    /** The selected object's label, highlighted in the list. */
    selectedLabelId: number | null;
    /** Jump to the first object using a label. */
    onSelect: (labelId: number) => void;
    /** Open the editor for a brand-new label. */
    onCreate: () => void;
    /** Open the editor for an existing label. */
    onEdit: (labelId: number) => void;
    /** Delete a label; its objects become unlabelled. */
    onDelete: (labelId: number) => void;
}

/**
 * The label table: one row per class, with how many objects use it.
 *
 * This is where a label is created, edited and removed. Rows keep list order,
 * because a label's position *is* its id — the number on its colour block.
 */
export function LabelsPanel({
    clip,
    selectedLabelId,
    onSelect,
    onCreate,
    onEdit,
    onDelete,
}: LabelsPanelProps) {
    const rows = useMemo(() => {
        const counts = new Map<number, number>();
        for (const tracklet of clip.tracklets) {
            if (tracklet.labelId === null) continue;
            counts.set(
                tracklet.labelId,
                (counts.get(tracklet.labelId) ?? 0) + 1,
            );
        }
        return clip.labels.map((label) => ({
            label,
            count: counts.get(label.id) ?? 0,
        }));
    }, [clip.labels, clip.tracklets]);

    return (
        <section className={styles.panel} aria-label="Labels">
            <header className={styles.head}>
                <span className="sectionLabel">Labels</span>
                <span className={styles.count}>{clip.labels.length}</span>
                <button
                    type="button"
                    className={styles.add}
                    onClick={onCreate}
                    aria-label="Create a label"
                    title="Create a label"
                >
                    <Icon name="plus" size={14} />
                </button>
            </header>

            <div className={`scrollArea ${styles.list}`}>
                {rows.length === 0 ? (
                    <EmptyState icon="layers" title="No labels yet">
                        Create one here, or pick one when you assign a label to
                        an object.
                    </EmptyState>
                ) : (
                    rows.map(({ label, count }) => {
                        const active = label.id === selectedLabelId;
                        return (
                            <div
                                key={label.id}
                                className={`${styles.row} ${
                                    active ? styles.selected : ""
                                }`}
                            >
                                <button
                                    type="button"
                                    className={styles.item}
                                    onClick={() => onSelect(label.id)}
                                    aria-current={active ? "true" : undefined}
                                    title={
                                        count > 0
                                            ? `Select an object labelled “${label.name}”`
                                            : "No object uses this label yet"
                                    }
                                >
                                    <span
                                        className={styles.block}
                                        style={{ background: label.color }}
                                    >
                                        {label.id}
                                    </span>
                                    <span className={styles.name}>
                                        {label.name}
                                    </span>
                                    <span className={styles.meta}>
                                        {count}{" "}
                                        {count === 1 ? "object" : "objects"}
                                    </span>
                                </button>
                                <button
                                    type="button"
                                    className={styles.action}
                                    onClick={() => onEdit(label.id)}
                                    aria-label={`Edit ${label.name}`}
                                    title="Edit this label"
                                >
                                    <Icon name="pencil" size={14} />
                                </button>
                                <button
                                    type="button"
                                    className={`${styles.action} ${styles.danger}`}
                                    onClick={() => onDelete(label.id)}
                                    aria-label={`Delete ${label.name}`}
                                    title={
                                        count > 0
                                            ? `Delete this label — its ${count} ${
                                                  count === 1
                                                      ? "object"
                                                      : "objects"
                                              } become unlabelled`
                                            : "Delete this label"
                                    }
                                >
                                    <Icon name="trash" size={14} />
                                </button>
                            </div>
                        );
                    })
                )}
            </div>
        </section>
    );
}
