import { useCallback, useMemo, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import { Button } from "../ui";
import styles from "./TimelineStrip.module.css";

/**
 * Per-frame state of one object, in the three marks the timeline uses.
 *
 * * `human` — a mask a person drew, corrected or cleared by hand. It is painted
 *   in the object's own (label) colour, and only a human frame may seed a run.
 * * `propagated` — a mask the tracker produced. It is painted orange on the strip
 *   as a standing reminder that nobody has checked it, while the mask itself is
 *   still drawn on the video in the label colour.
 * * `none` — no mask. The strip's background is what "the tracker found nothing
 *   here" looks like.
 */
export type TimelineFrameState = "none" | "propagated" | "human";

/** How one frame's state is painted inside the strip's 0..1 tall viewBox. */
interface CellPaint {
    fill: string;
    opacity: number;
}

const NEUTRAL = "#20262d";
const OBJECT_FALLBACK = "#5c8fc9";
/** The one colour reserved for "a machine produced this, not a person". */
const PROPAGATED = "#e2803a";

/** The paint for one state. `colour` is the selected object's own colour. */
function paint(state: TimelineFrameState, colour: string): CellPaint {
    switch (state) {
        case "human":
            return { fill: colour, opacity: 1 };
        case "propagated":
            return { fill: PROPAGATED, opacity: 1 };
        default:
            return { fill: NEUTRAL, opacity: 1 };
    }
}

const LEGEND: { state: TimelineFrameState; label: string }[] = [
    { state: "human", label: "human" },
    { state: "propagated", label: "propagated" },
];

export interface TimelineStripProps {
    frameCount: number;
    frameIndex: number;
    /** One state per frame, indexed by frame. */
    states: TimelineFrameState[];
    /** The selected object's colour; human frames are painted with it. */
    color?: string;
    /** What the strip is showing, for the caption. */
    label?: string;
    /**
     * The propagation range, as inclusive frame indices.
     *
     * When present the strip holds back the frames outside it, so the range the
     * next run will cover is read directly from the track.
     */
    rangeFirst?: number;
    rangeLast?: number;
    /** The frame carrying the mask; the range never excludes it. */
    rangeAnchor?: number;
    /**
     * Called when a handle is dragged. Its presence is what shows the two bars;
     * a read-only range (during a run or a review) passes nothing and only the
     * band is drawn.
     */
    onRangeChange?: (first: number, last: number) => void;
    /** Selecting a populated run opens a staged trim/delete edit. */
    onClipSelect?: (first: number, last: number) => void;
    clipEdit?: {
        first: number;
        last: number;
        originalFirst: number;
        originalLast: number;
        deleted: boolean;
    };
    onClipRangeChange?: (first: number, last: number) => void;
    onClipDeleteToggle?: (deleted: boolean) => void;
    onClipConfirm?: () => void;
    onClipCancel?: () => void;
    canConfirmClipEdit?: boolean;
    onPromoteCurrent?: () => void;
}

type RangeHandle = "first" | "last";

/** Width of a range bar, in pixels; also the inset that keeps one on-screen. */
const RANGE_HANDLE_W = 9;

export function TimelineStrip(props: TimelineStripProps) {
    const {
        frameCount,
        frameIndex,
        states,
        color,
        label,
        rangeFirst,
        rangeLast,
        rangeAnchor,
        onRangeChange,
        onClipSelect,
        clipEdit,
        onClipRangeChange,
        onClipDeleteToggle,
        onClipConfirm,
        onClipCancel,
        canConfirmClipEdit = false,
        onPromoteCurrent,
    } = props;
    const trackRef = useRef<HTMLDivElement>(null);
    const [rangeDrag, setRangeDrag] = useState<RangeHandle | null>(null);
    const [clipDrag, setClipDrag] = useState<RangeHandle | null>(null);

    const objectColour = color ?? OBJECT_FALLBACK;
    const hasRange = rangeFirst !== undefined && rangeLast !== undefined;
    const anchor = rangeAnchor ?? frameIndex;

    /**
     * Adjacent frames in the same state collapse into one rectangle.
     *
     * A clip can hold thousands of frames, and one DOM node per frame would make
     * the track sluggish for no visual gain: a run of identical states is one mark.
     */
    const runs = useMemo(() => {
        const out: {
            state: TimelineFrameState;
            start: number;
            length: number;
        }[] = [];
        for (let frame = 0; frame < states.length; frame++) {
            const state = states[frame];
            const last = out[out.length - 1];
            if (last && last.state === state) last.length += 1;
            else out.push({ state, start: frame, length: 1 });
        }
        return out;
    }, [states]);

    const counts = useMemo(() => {
        const tally: Partial<Record<TimelineFrameState, number>> = {};
        for (const state of states) tally[state] = (tally[state] ?? 0) + 1;
        return tally;
    }, [states]);

    /** Map a pointer position onto a frame index. */
    const frameAt = useCallback(
        (clientX: number): number => {
            const rect = trackRef.current?.getBoundingClientRect();
            if (!rect || rect.width <= 0) return 0;
            const ratio = (clientX - rect.left) / rect.width;
            return Math.max(
                0,
                Math.min(frameCount - 1, Math.floor(ratio * frameCount)),
            );
        },
        [frameCount],
    );

    const selectRunAt = useCallback(
        (frame: number) => {
            if (!onClipSelect || states[frame] === "none") return;
            let first = frame;
            let last = frame;
            while (first > 0 && states[first - 1] !== "none") first -= 1;
            while (
                last < states.length - 1 &&
                states[last + 1] !== "none"
            )
                last += 1;
            onClipSelect(first, last);
        },
        [onClipSelect, states],
    );

    const onTrackPointerDown = useCallback(
        (event: ReactPointerEvent<HTMLDivElement>) => {
            if (event.button !== 0) return;
            selectRunAt(frameAt(event.clientX));
        },
        [frameAt, selectRunAt],
    );

    /** Map a frame index onto the strip's horizontal percentage. */
    const percent = useCallback(
        (frame: number) => (frameCount ? (frame / frameCount) * 100 : 0),
        [frameCount],
    );

    const startRangeDrag = useCallback(
        (handle: RangeHandle) => (event: ReactPointerEvent<HTMLDivElement>) => {
            if (!onRangeChange) return;
            event.preventDefault();
            event.stopPropagation();
            event.currentTarget.setPointerCapture?.(event.pointerId);
            setRangeDrag(handle);
        },
        [onRangeChange],
    );

    const moveRangeHandle = useCallback(
        (handle: RangeHandle) => (event: ReactPointerEvent<HTMLDivElement>) => {
            if (rangeDrag !== handle || !onRangeChange) return;
            const frame = frameAt(event.clientX);
            if (handle === "first") {
                // Never past the anchor: it carries the mask and a range that
                // excluded it could not seed a run.
                onRangeChange(Math.min(frame, anchor), rangeLast as number);
            } else {
                onRangeChange(rangeFirst as number, Math.max(frame, anchor));
            }
        },
        [rangeDrag, onRangeChange, frameAt, anchor, rangeFirst, rangeLast],
    );

    const endRangeDrag = useCallback(
        (event: ReactPointerEvent<HTMLDivElement>) => {
            event.currentTarget.releasePointerCapture?.(event.pointerId);
            setRangeDrag(null);
        },
        [],
    );

    const startClipDrag = useCallback(
        (handle: RangeHandle) => (event: ReactPointerEvent<HTMLDivElement>) => {
            if (!onClipRangeChange || !clipEdit || clipEdit.deleted) return;
            event.preventDefault();
            event.stopPropagation();
            event.currentTarget.setPointerCapture?.(event.pointerId);
            setClipDrag(handle);
        },
        [onClipRangeChange, clipEdit],
    );

    const moveClipHandle = useCallback(
        (handle: RangeHandle) => (event: ReactPointerEvent<HTMLDivElement>) => {
            if (clipDrag !== handle || !onClipRangeChange || !clipEdit) return;
            const frame = frameAt(event.clientX);
            if (handle === "first") {
                onClipRangeChange(
                    Math.max(
                        clipEdit.originalFirst,
                        Math.min(frame, clipEdit.last - 1),
                    ),
                    clipEdit.last,
                );
            } else {
                onClipRangeChange(
                    clipEdit.first,
                    Math.min(
                        clipEdit.originalLast,
                        Math.max(frame, clipEdit.first + 1),
                    ),
                );
            }
        },
        [clipDrag, onClipRangeChange, clipEdit, frameAt],
    );

    const endClipDrag = useCallback(
        (event: ReactPointerEvent<HTMLDivElement>) => {
            event.currentTarget.releasePointerCapture?.(event.pointerId);
            setClipDrag(null);
        },
        [],
    );

    if (frameCount <= 0) return null;

    const playheadPercent = ((frameIndex + 0.5) / frameCount) * 100;

    return (
        <div className={styles.wrap}>
            <div className={styles.caption}>
                <span className={styles.captionLabel}>
                    {label ?? "Timeline"}
                </span>
                {clipEdit && onClipCancel && onClipConfirm && (
                    <span className={styles.actions}>
                        <span className={styles.editLabel}>
                            Frames {clipEdit.first + 1}–{clipEdit.last + 1}
                        </span>
                        {onClipDeleteToggle && (
                            <Button
                                size="sm"
                                variant={clipEdit.deleted ? "default" : "danger"}
                                icon={clipEdit.deleted ? "undo" : "trash"}
                                onClick={() =>
                                    onClipDeleteToggle(!clipEdit.deleted)
                                }
                            >
                                {clipEdit.deleted ? "Keep clip" : "Delete clip"}
                            </Button>
                        )}
                        <Button
                            size="sm"
                            variant="primary"
                            icon="check"
                            disabled={!canConfirmClipEdit}
                            onClick={onClipConfirm}
                        >
                            {clipEdit.deleted
                                ? "Confirm delete"
                                : "Confirm edit"}
                        </Button>
                        <Button size="sm" onClick={onClipCancel}>
                            Cancel
                        </Button>
                    </span>
                )}
                <span className={styles.legend}>
                    {LEGEND.map((item) => {
                        const value = counts[item.state] ?? 0;
                        if (value === 0) return null;
                        const cell = paint(item.state, objectColour);
                        return (
                            <span
                                key={item.state}
                                className={styles.legendItem}
                            >
                                <span
                                    className={styles.swatch}
                                    style={{
                                        background: cell.fill,
                                        opacity: cell.opacity,
                                    }}
                                />
                                {item.label} {value}
                            </span>
                        );
                    })}
                </span>
                {!clipEdit && onPromoteCurrent && (
                    <Button
                        size="sm"
                        icon="check"
                        title="Mark the current mask as human-annotated so it can seed propagation"
                        onClick={onPromoteCurrent}
                    >
                        Mark current mask human
                    </Button>
                )}
            </div>

            <div
                ref={trackRef}
                className={styles.track}
                role={onClipSelect ? "group" : "img"}
                tabIndex={onClipSelect ? 0 : undefined}
                aria-label={`${label ?? "Timeline"} — current frame ${frameIndex + 1} of ${frameCount}. Click a colored segment to edit it; use the bottom controls to seek.`}
                onPointerDown={onTrackPointerDown}
                onKeyDown={(event) => {
                    if (
                        (event.key === "Enter" || event.key === " ") &&
                        onClipSelect
                    ) {
                        event.preventDefault();
                        selectRunAt(frameIndex);
                    }
                }}
            >
                <svg
                    className={styles.svg}
                    viewBox={`0 0 ${Math.max(1, frameCount)} 1`}
                    preserveAspectRatio="none"
                    aria-hidden="true"
                >
                    {runs.map((run) => {
                        const cell = paint(run.state, objectColour);
                        return (
                            <rect
                                key={run.start}
                                x={run.start}
                                y={0}
                                width={run.length}
                                height={1}
                                fill={cell.fill}
                                opacity={cell.opacity}
                            />
                        );
                    })}
                </svg>
                {hasRange && (
                    <>
                        {/* Frames outside the range are held back. */}
                        {rangeFirst! > 0 && (
                            <div
                                className={styles.rangeShade}
                                style={{
                                    left: 0,
                                    width: `${percent(rangeFirst!)}%`,
                                }}
                            />
                        )}
                        {rangeLast! < frameCount - 1 && (
                            <div
                                className={styles.rangeShade}
                                style={{
                                    left: `${percent(rangeLast! + 1)}%`,
                                    right: 0,
                                }}
                            />
                        )}
                        {/* The edges of the range. The cells are left alone: a
                            human frame must keep its own label colour. */}
                        <div
                            className={styles.rangeBand}
                            style={{
                                left: `${percent(rangeFirst!)}%`,
                                width: `${percent(rangeLast! + 1) - percent(rangeFirst!)}%`,
                            }}
                        />
                        <div
                            className={styles.rangeAnchor}
                            style={{ left: `${percent(anchor + 0.5)}%` }}
                            title={`Anchor: frame ${anchor + 1}`}
                        />
                        {onRangeChange &&
                            (["first", "last"] as RangeHandle[]).map(
                                (handle) => (
                                    <div
                                        key={handle}
                                        role="slider"
                                        aria-label={
                                            handle === "first"
                                                ? "First frame of the propagation range"
                                                : "Last frame of the propagation range"
                                        }
                                        aria-valuemin={0}
                                        aria-valuemax={frameCount - 1}
                                        aria-valuenow={
                                            handle === "first"
                                                ? rangeFirst
                                                : rangeLast
                                        }
                                        tabIndex={0}
                                        className={`${styles.rangeHandle} ${rangeDrag === handle ? styles.rangeHandleActive : ""}`}
                                        style={
                                            handle === "first"
                                                ? {
                                                      left: `${percent(rangeFirst!)}%`,
                                                  }
                                                : {
                                                      left: `calc(${percent(rangeLast! + 1)}% - ${RANGE_HANDLE_W}px)`,
                                                  }
                                        }
                                        title={`${handle === "first" ? "First" : "Last"} frame of the range: ${
                                            (handle === "first"
                                                ? rangeFirst!
                                                : rangeLast!) + 1
                                        }. Drag to resize.`}
                                        onPointerDown={startRangeDrag(handle)}
                                        onPointerMove={moveRangeHandle(handle)}
                                        onPointerUp={endRangeDrag}
                                        onPointerCancel={endRangeDrag}
                                        onKeyDown={(event) => {
                                            if (!onRangeChange) return;
                                            const step = event.shiftKey
                                                ? 10
                                                : 1;
                                            const delta =
                                                event.key === "ArrowLeft"
                                                    ? -step
                                                    : event.key === "ArrowRight"
                                                      ? step
                                                      : 0;
                                            if (!delta) return;
                                            event.preventDefault();
                                            if (handle === "first") {
                                                onRangeChange(
                                                    Math.max(
                                                        0,
                                                        Math.min(
                                                            rangeFirst! + delta,
                                                            anchor,
                                                        ),
                                                    ),
                                                    rangeLast!,
                                                );
                                            } else {
                                                onRangeChange(
                                                    rangeFirst!,
                                                    Math.min(
                                                        frameCount - 1,
                                                        Math.max(
                                                            rangeLast! + delta,
                                                            anchor,
                                                        ),
                                                    ),
                                                );
                                            }
                                        }}
                                    />
                                ),
                            )}
                    </>
                )}
                {clipEdit && (
                    <>
                        {clipEdit.deleted ? (
                            <div
                                className={`${styles.clipRemoved} ${styles.clipRemovedAll}`}
                                style={{
                                    left: `${percent(clipEdit.originalFirst)}%`,
                                    width: `${percent(clipEdit.originalLast + 1) - percent(clipEdit.originalFirst)}%`,
                                }}
                            />
                        ) : (
                            <>
                                {clipEdit.first > clipEdit.originalFirst && (
                                    <div
                                        className={styles.clipRemoved}
                                        style={{
                                            left: `${percent(clipEdit.originalFirst)}%`,
                                            width: `${percent(clipEdit.first) - percent(clipEdit.originalFirst)}%`,
                                        }}
                                    />
                                )}
                                {clipEdit.last < clipEdit.originalLast && (
                                    <div
                                        className={styles.clipRemoved}
                                        style={{
                                            left: `${percent(clipEdit.last + 1)}%`,
                                            width: `${percent(clipEdit.originalLast + 1) - percent(clipEdit.last + 1)}%`,
                                        }}
                                    />
                                )}
                                <div
                                    className={styles.clipBand}
                                    style={{
                                        left: `${percent(clipEdit.first)}%`,
                                        width: `${percent(clipEdit.last + 1) - percent(clipEdit.first)}%`,
                                    }}
                                />
                            </>
                        )}
                        {!clipEdit.deleted &&
                            onClipRangeChange &&
                            (["first", "last"] as RangeHandle[]).map(
                                (handle) => (
                                    <div
                                        key={handle}
                                        role="slider"
                                        aria-label={
                                            handle === "first"
                                                ? "First frame of the clip"
                                                : "Last frame of the clip"
                                        }
                                        aria-valuemin={
                                            clipEdit.originalFirst + 1
                                        }
                                        aria-valuemax={
                                            clipEdit.originalLast + 1
                                        }
                                        aria-valuenow={
                                            (handle === "first"
                                                ? clipEdit.first
                                                : clipEdit.last) + 1
                                        }
                                        tabIndex={0}
                                        className={`${styles.clipHandle} ${clipDrag === handle ? styles.rangeHandleActive : ""}`}
                                        style={
                                            handle === "first"
                                                ? {
                                                      left: `${percent(clipEdit.first)}%`,
                                                  }
                                                : {
                                                      left: `calc(${percent(clipEdit.last + 1)}% - ${RANGE_HANDLE_W}px)`,
                                                  }
                                        }
                                        title={`Drag to trim the ${handle === "first" ? "start" : "end"} of the selected clip`}
                                        onPointerDown={startClipDrag(handle)}
                                        onPointerMove={moveClipHandle(handle)}
                                        onPointerUp={endClipDrag}
                                        onPointerCancel={endClipDrag}
                                        onKeyDown={(event) => {
                                            if (!onClipRangeChange) return;
                                            const delta =
                                                event.key === "ArrowLeft"
                                                    ? -1
                                                    : event.key === "ArrowRight"
                                                      ? 1
                                                      : 0;
                                            if (!delta) return;
                                            event.preventDefault();
                                            if (handle === "first") {
                                                onClipRangeChange(
                                                    Math.max(
                                                        clipEdit.originalFirst,
                                                        Math.min(
                                                            clipEdit.first +
                                                                delta,
                                                            clipEdit.last - 1,
                                                        ),
                                                    ),
                                                    clipEdit.last,
                                                );
                                            } else {
                                                onClipRangeChange(
                                                    clipEdit.first,
                                                    Math.min(
                                                        clipEdit.originalLast,
                                                        Math.max(
                                                            clipEdit.last +
                                                                delta,
                                                            clipEdit.first + 1,
                                                        ),
                                                    ),
                                                );
                                            }
                                        }}
                                    />
                                ),
                            )}
                    </>
                )}
                <div
                    className={styles.playhead}
                    style={{ left: `${playheadPercent}%` }}
                />
                <div
                    className={styles.playheadHandle}
                    style={{ left: `${playheadPercent}%` }}
                />
            </div>
        </div>
    );
}
