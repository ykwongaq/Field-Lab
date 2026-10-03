import { useMemo, useState } from "react";
import type { Clip } from "../lib/clip";
import { MODE_VOCABULARY } from "../lib/project";
import { EmptyState, Icon } from "../ui";
import { LabelPicker } from "./LabelPicker";
import styles from "./TrackletList.module.css";

interface TrackletListProps {
    clip: Clip;
    selectedId: number | null;
    onSelect: (id: number) => void;
    /** Ids queued for propagation. */
    batch?: number[];
    /**
     * Shift-click handler.
     *
     * Only passed in while the propagate tool is open: queueing objects is a
     * propagation concept, so the modifier means nothing in the other tools.
     */
    onToggleBatch?: (id: number) => void;
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
 * draws. Each row's colour block is also its label button: it shows the label's
 * colour and id, and opens the picker that changes it.
 */
export function TrackletList({
    clip,
    selectedId,
    onSelect,
    batch = [],
    onToggleBatch,
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
                        const queued = batch.includes(tracklet.id);
                        const active = tracklet.id === selectedId;
                        return (
                            <div
                                key={tracklet.id}
                                className={`${styles.row} ${active ? styles.selected : ""}`}
                            >
                                {onAssign && (
                                    <button
                                        type="button"
                                        className={styles.assign}
                                        style={{ background: tracklet.color }}
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
                                        aria-label={`Assign a label to ${vocab.unit} #${tracklet.id}. Current: ${tracklet.label}`}
                                        title="Assign a label"
                                    >
                                        {tracklet.labelId ?? "–"}
                                    </button>
                                )}
                                <button
                                    type="button"
                                    className={styles.item}
                                    onClick={(event) =>
                                        event.shiftKey && onToggleBatch
                                            ? onToggleBatch(tracklet.id)
                                            : onSelect(tracklet.id)
                                    }
                                    aria-current={active ? "true" : undefined}
                                    title={
                                        onToggleBatch
                                            ? "Shift-click to queue this one for propagation"
                                            : undefined
                                    }
                                >
                                    <span className={styles.body}>
                                        <span className={styles.label}>
                                            {tracklet.label}
                                        </span>
                                        <span className={styles.meta}>
                                            #{tracklet.id}
                                            {vocab.hasObjectIdentity
                                                ? ` · obj ${tracklet.objectId}`
                                                : ""}{" "}
                                            · {tracklet.maskFrames.count} frames
                                        </span>
                                    </span>
                                    {queued && (
                                        <span
                                            className="chip chipAccent"
                                            title="Queued for propagation; Shift-click to remove from the queue"
                                        >
                                            queued
                                        </span>
                                    )}
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
