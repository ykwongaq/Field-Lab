import { useMemo, useState } from "react";
import type { Clip } from "../lib/clip";
import { MODE_VOCABULARY } from "../lib/project";
import { EmptyState, Icon } from "../ui";
import { LabelPicker } from "./LabelPicker";
import styles from "./TrackletList.module.css";

interface TrackletListProps {
    clip: Clip;
    /** The primary selection: the row Add/Edit eventually acts on. */
    selectedId: number | null;
    /** Every selected row, including the primary. */
    selectedIds?: number[];
    onSelect: (id: number) => void;
    /** Toggle membership without moving the playhead. */
    onToggleSelect?: (id: number) => void;
    /** True while a run owns the selection. */
    disabled?: boolean;
    /**
     * Delete handler.
     *
     * When passed, each row grows a trash button that appears on hover and on
     * the selected row. Removal is confirmed by the caller, not here.
     */
    onDelete?: (id: number) => void;
    /** Assign a label to one object (instance mode). */
    onAssign?: (trackletId: number, labelId: number | null) => void;
    /** Create a new label and assign it to one object (instance mode). */
    onNewLabel?: (trackletId: number) => void;
}

/**
 * The object list: every tracklet in the clip, searchable.
 *
 * It is the workspace's index — selecting a row decides which mask the canvas
 * draws. Each row leads with the object's own id, then its label: a colour
 * swatch (its label's colour) that opens the picker, and the label's name.
 * The label's *number* is deliberately not shown here — a whole species shares
 * one label id, so leading with it made every row of "Marine Animals" read as
 * `0`. That number belongs to the label table, not to the object.
 */
export function TrackletList({
    clip,
    selectedId,
    selectedIds = [],
    onSelect,
    onToggleSelect,
    disabled = false,
    onDelete,
    onAssign,
    onNewLabel,
}: TrackletListProps) {
    const [query, setQuery] = useState("");
    const [picker, setPicker] = useState<{
        trackletId: number;
        anchor: DOMRect;
        current: number | null;
    } | null>(null);
    const vocab = MODE_VOCABULARY[clip.mode];

    const items = useMemo(() => {
        const needle = query.trim().toLowerCase();
        if (!needle) return clip.tracklets;
        return clip.tracklets.filter(
            (tracklet) =>
                tracklet.label.toLowerCase().includes(needle) ||
                String(tracklet.id).includes(needle) ||
                String(tracklet.objectId).includes(needle),
        );
    }, [clip.tracklets, query]);

    return (
        <section className={styles.panel} aria-label={`${vocab.units} list`}>
            <header className={styles.head}>
                <span className="sectionLabel">{vocab.units}</span>
                <span className={styles.count}>
                    {query.trim()
                        ? `${items.length} / ${clip.tracklets.length}`
                        : clip.tracklets.length}
                </span>
            </header>

            <div className={styles.searchWrap}>
                <Icon name="search" size={14} className={styles.searchIcon} />
                <input
                    type="search"
                    className={`input ${styles.search}`}
                    placeholder={`Filter ${vocab.units}…`}
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                    aria-label={`Filter ${vocab.units}`}
                />
            </div>

            <div className={`scrollArea ${styles.list}`}>
                {clip.tracklets.length === 0 ? (
                    <EmptyState icon="layers" title={`No ${vocab.units} yet`}>
                        Draw a mask on a frame to create the first one.
                    </EmptyState>
                ) : items.length === 0 ? (
                    <EmptyState icon="search" title="No matches">
                        Nothing here matches “{query.trim()}”.
                    </EmptyState>
                ) : (
                    items.map((tracklet) => {
                        const checked = selectedIds.includes(tracklet.id);
                        const active = tracklet.id === selectedId;
                        return (
                            <div
                                key={tracklet.id}
                                className={`${styles.row} ${active ? styles.selected : ""} ${
                                    checked && !active ? styles.checked : ""
                                }`}
                            >
                                {onToggleSelect && (
                                    <input
                                        type="checkbox"
                                        className={styles.check}
                                        checked={checked}
                                        disabled={disabled}
                                        onChange={() =>
                                            onToggleSelect(tracklet.id)
                                        }
                                        aria-label={`Select ${vocab.unit} #${tracklet.id} for a multi-selection`}
                                        title={
                                            disabled
                                                ? "Selection is locked while a run is live"
                                                : "Add to or remove from the selection"
                                        }
                                    />
                                )}
                                <span
                                    className={styles.idTag}
                                    title={`${vocab.unit} #${tracklet.id}`}
                                >
                                    #{tracklet.id}
                                </span>
                                {onAssign && (
                                    <button
                                        type="button"
                                        className={styles.assign}
                                        disabled={disabled}
                                        onClick={(event) => {
                                            const anchor =
                                                event.currentTarget.getBoundingClientRect();
                                            setPicker((current) =>
                                                current?.trackletId ===
                                                tracklet.id
                                                    ? null
                                                    : {
                                                          trackletId:
                                                              tracklet.id,
                                                          anchor,
                                                          current:
                                                              tracklet.labelId,
                                                      },
                                            );
                                        }}
                                        aria-label={`Change the label for ${vocab.unit} #${tracklet.id}. Current: ${tracklet.label}`}
                                        title={`Label: ${tracklet.label} — click to change`}
                                    >
                                        <span
                                            className={styles.swatch}
                                            style={{
                                                background: tracklet.color,
                                            }}
                                        />
                                    </button>
                                )}
                                <button
                                    type="button"
                                    className={styles.item}
                                    disabled={disabled}
                                    onClick={(event) => {
                                        if (
                                            (event.ctrlKey || event.metaKey) &&
                                            onToggleSelect
                                        ) {
                                            onToggleSelect(tracklet.id);
                                            return;
                                        }
                                        onSelect(tracklet.id);
                                    }}
                                    aria-current={active ? "true" : undefined}
                                    title={
                                        disabled
                                            ? "Selection is locked while a run is live"
                                            : onToggleSelect
                                              ? "Ctrl/Cmd-click to add to the selection"
                                              : undefined
                                    }
                                >
                                    <span className={styles.body}>
                                        <span
                                            className={`${styles.label} ${
                                                tracklet.labelId === null
                                                    ? styles.unlabelled
                                                    : ""
                                            }`}
                                        >
                                            {tracklet.label}
                                        </span>
                                        <span className={styles.meta}>
                                            {vocab.hasObjectIdentity
                                                ? `obj ${tracklet.objectId} · `
                                                : ""}
                                            {tracklet.maskFrames.count} frames
                                        </span>
                                    </span>
                                </button>
                                {onDelete && (
                                    <button
                                        type="button"
                                        className={styles.trash}
                                        onClick={() => onDelete(tracklet.id)}
                                        aria-label={`Delete ${vocab.unit} #${tracklet.id}`}
                                        title={`Delete ${vocab.unit} #${tracklet.id}`}
                                    >
                                        <Icon name="trash" size={14} />
                                    </button>
                                )}
                            </div>
                        );
                    })
                )}
            </div>

            {picker && onAssign && (
                <LabelPicker
                    anchor={picker.anchor}
                    labels={clip.labels}
                    current={picker.current}
                    onPick={(labelId) => {
                        onAssign(picker.trackletId, labelId);
                        setPicker(null);
                    }}
                    onNew={() => {
                        onNewLabel?.(picker.trackletId);
                        setPicker(null);
                    }}
                    onClose={() => setPicker(null)}
                />
            )}
        </section>
    );
}
