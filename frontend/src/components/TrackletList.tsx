import { useMemo, useState } from "react";
import type { Clip } from "../lib/clip";
import { MODE_VOCABULARY } from "../lib/project";
import { EmptyState, Icon } from "../ui";
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
}

/**
 * The object list: every tracklet in the clip, searchable.
 *
 * It is the workspace's index — selecting a row decides which mask the canvas
 * draws and which taxonomy the inspector edits.
 */
export function TrackletList({
    clip,
    selectedId,
    onSelect,
    batch = [],
    onToggleBatch,
}: TrackletListProps) {
    const [query, setQuery] = useState("");
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
                            <button
                                key={tracklet.id}
                                type="button"
                                className={`${styles.item} ${active ? styles.selected : ""}`}
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
                                <span
                                    className={styles.swatch}
                                    style={{ background: tracklet.color }}
                                />
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
                        );
                    })
                )}
            </div>
        </section>
    );
}
