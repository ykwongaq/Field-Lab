import { useCallback, useMemo, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import styles from "./TimelineStrip.module.css";

/**
 * Per-frame state of one object.
 *
 * The first five are the spec's §8.1 frame states (`MASK_INTERACTION_SPEC.md`);
 * the last two are the refinements this build needs on top of them:
 *
 * * `verified` — an `accepted` frame that a human drew, corrected or cleared by
 *   hand. It reads as solid, against the translucent `accepted` a bulk accept
 *   leaves behind, because only a verified frame may anchor a second run.
 * * `preview` — a mask a propagation run has produced that is not committed yet,
 *   which the spec calls a "propagated draft".
 *
 * `lost` is the tracker reporting no object on a frame it covered (occlusion or
 * off-screen). It is derived, not stored: a frame the finished run never produced
 * a mask for is a frame it did not find the object on.
 */
export type TimelineFrameState =
    | "none"
    | "accepted"
    | "verified"
    | "draft"
    | "preview"
    | "stale"
    | "lost";

/** How one frame's state is painted inside the strip's 0..1 tall viewBox. */
interface CellPaint {
    fill: string;
    opacity: number;
    /** Vertical band, so `lost` can read as a short dash rather than a block. */
    y: number;
    height: number;
}

const NEUTRAL = "#e9eef3";
const OBJECT_FALLBACK = "#4a7fb5";
const DRAFT = "#f0b429";
const PREVIEW = "#f6cf6b";
const STALE = "#8fa3b8";
const LOST = "#7f8c8d";
const BOUNDARY = "#5b6b7b";

/** The paint for one state. `colour` is the selected object's own colour. */
function paint(state: TimelineFrameState, colour: string): CellPaint {
    switch (state) {
        case "accepted":
            // Committed in bulk: the object's colour, held back so a verified
            // frame can read as the stronger mark it is.
            return { fill: colour, opacity: 0.72, y: 0, height: 1 };
        case "verified":
            return { fill: colour, opacity: 1, y: 0, height: 1 };
        case "draft":
            return { fill: DRAFT, opacity: 1, y: 0, height: 1 };
        case "preview":
            return { fill: PREVIEW, opacity: 1, y: 0, height: 1 };
        case "stale":
            return { fill: STALE, opacity: 0.6, y: 0, height: 1 };
        case "lost":
            return { fill: LOST, opacity: 1, y: 0.35, height: 0.3 };
        default:
            return { fill: NEUTRAL, opacity: 1, y: 0, height: 1 };
    }
}

const LEGEND: { state: TimelineFrameState; label: string }[] = [
    { state: "accepted", label: "accepted" },
    { state: "verified", label: "verified" },
    { state: "draft", label: "draft" },
    { state: "stale", label: "stale" },
    { state: "lost", label: "not found" },
];

export interface TimelineStripProps {
    frameCount: number;
    frameIndex: number;
    /** One state per frame, indexed by frame. */
    states: TimelineFrameState[];
    /** The selected object's colour; accepted frames are painted with it. */
    color?: string;
    /**
     * Window boundaries of the running job, as frame indices.
     *
     * A long run is chunked into overlapping windows, and the hand-off between
     * them is where a track can drift, so the boundaries are worth showing.
     */
    boundaries?: number[];
    /** What the strip is showing, for the caption. */
    label?: string;
    onSeek: (frame: number) => void;
}

export function TimelineStrip(props: TimelineStripProps) {
    const {
        frameCount,
        frameIndex,
        states,
        color,
        boundaries,
        label,
        onSeek,
    } = props;
    const trackRef = useRef<HTMLDivElement>(null);
    const [dragging, setDragging] = useState(false);

    const objectColour = color ?? OBJECT_FALLBACK;

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
            if (!dragging) return;
            onSeek(frameAt(event.clientX));
        },
        [dragging, frameAt, onSeek],
    );

    const endDrag = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
        event.currentTarget.releasePointerCapture?.(event.pointerId);
        setDragging(false);
    }, []);

    if (frameCount <= 0) return null;

    const playheadPercent = ((frameIndex + 0.5) / frameCount) * 100;
    // A boundary tick should read as a hairline: roughly a pixel at the width the
    // strip is usually drawn at, in the viewBox's frame units.
    const tickWidth = Math.max(1, frameCount / 400);

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
                                y={cell.y}
                                width={run.length}
                                height={cell.height}
                                fill={cell.fill}
                                opacity={cell.opacity}
                            />
                        );
                    })}
                    {(boundaries ?? []).map((frame) =>
                        frame > 0 && frame < frameCount ? (
                            <rect
                                key={`boundary-${frame}`}
                                x={frame - tickWidth / 2}
                                y={0}
                                width={tickWidth}
                                height={1}
                                fill={BOUNDARY}
                                opacity={0.45}
                            />
                        ) : null,
                    )}
                </svg>
                <div
                    className={styles.playhead}
                    style={{ left: `${playheadPercent}%` }}
                />
            </div>
        </div>
    );
}
