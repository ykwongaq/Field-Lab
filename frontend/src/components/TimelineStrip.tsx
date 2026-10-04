import { useCallback, useMemo, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
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
     * next run will cover is read off the same band the reviewer scrubs on
     * instead of typed into a separate form.
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
    onSeek: (frame: number) => void;
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
        onSeek,
    } = props;
    const trackRef = useRef<HTMLDivElement>(null);
    const [dragging, setDragging] = useState(false);
    const [rangeDrag, setRangeDrag] = useState<RangeHandle | null>(null);

    const objectColour = color ?? OBJECT_FALLBACK;
    const hasRange = rangeFirst !== undefined && rangeLast !== undefined;
    const anchor = rangeAnchor ?? frameIndex;

    /**
     * Adjacent frames in the same state collapse into one rectangle.
     *
     * A clip can hold thousands of frames, and one DOM node per frame would make
     * scrubbing janky for no visual gain: a run of identical states is one mark.
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

    const onPointerDown = useCallback(
        (event: ReactPointerEvent<HTMLDivElement>) => {
            if (event.button !== 0) return;
            event.currentTarget.setPointerCapture(event.pointerId);
            setDragging(true);
            onSeek(frameAt(event.clientX));
        },
        [frameAt, onSeek],
    );

    const onPointerMove = useCallback(
        (event: ReactPointerEvent<HTMLDivElement>) => {
            // A handle has the pointer; seeking now would fight the drag.
            if (rangeDrag || !dragging) return;
            onSeek(frameAt(event.clientX));
        },
        [rangeDrag, dragging, frameAt, onSeek],
    );

    const endDrag = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
        event.currentTarget.releasePointerCapture?.(event.pointerId);
        setDragging(false);
    }, []);

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

    if (frameCount <= 0) return null;

    const playheadPercent = ((frameIndex + 0.5) / frameCount) * 100;

    return (
        <div className={styles.wrap}>
            <div className={styles.caption}>
                <span className={styles.captionLabel}>
                    {label ?? "Timeline"}
                </span>
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
            </div>

            <div
                ref={trackRef}
                className={styles.track}
                role="slider"
                tabIndex={0}
                aria-label={`${label ?? "Timeline"} — frame ${frameIndex + 1} of ${frameCount}`}
                aria-valuemin={1}
                aria-valuemax={frameCount}
                aria-valuenow={frameIndex + 1}
                onPointerDown={onPointerDown}
                onPointerMove={onPointerMove}
                onPointerUp={endDrag}
                onPointerCancel={endDrag}
                onKeyDown={(event) => {
                    if (event.key === "ArrowLeft")
                        onSeek(Math.max(0, frameIndex - 1));
                    else if (event.key === "ArrowRight")
                        onSeek(Math.min(frameCount - 1, frameIndex + 1));
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
