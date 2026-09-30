import { useCallback, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";

/**
 * The propagation range, as a bar over the whole clip.
 *
 * Two handles set the frame range and the anchor stays inside it — the anchor is
 * the frame carrying the mask being propagated, so a range that excluded it could
 * not run at all. The bar also separates what was *asked for* from what the run
 * will actually *produce*: a forward run over a range that starts before the
 * anchor never touches those earlier frames, and showing that difference here is
 * cheaper than explaining a progress bar that stops short of 100 %.
 *
 * It is an input, not the state: the workspace keeps the range (and the precise
 * "N frames before/after" fields) and this only reports changes.
 */

export interface PropagationRangeProps {
    frameCount: number;
    /** The frame whose mask is propagated; always inside `[first, last]`. */
    anchor: number;
    first: number;
    last: number;
    direction: "forward" | "backward" | "both";
    /** Frames the running job has produced so far. */
    produced?: ReadonlySet<number> | null;
    /** True while a job is running, so the bar reads as busy. */
    running?: boolean;
    onChange: (first: number, last: number) => void;
}

type Handle = "first" | "last";

const HANDLE_WIDTH = 11;

export function PropagationRange(props: PropagationRangeProps) {
    const trackRef = useRef<HTMLDivElement>(null);
    const [dragging, setDragging] = useState<Handle | null>(null);

    const lastFrame = Math.max(0, props.frameCount - 1);
    const percent = useCallback(
        (frame: number) => (lastFrame ? (frame / lastFrame) * 100 : 0),
        [lastFrame],
    );

    /** Map a pointer position onto the nearest frame. */
    const frameAt = useCallback(
        (clientX: number): number => {
            const rect = trackRef.current?.getBoundingClientRect();
            if (!rect || rect.width <= 0) return props.anchor;
            const ratio = (clientX - rect.left) / rect.width;
            const clamped = Math.min(1, Math.max(0, ratio));
            return Math.round(clamped * lastFrame);
        },
        [lastFrame, props.anchor],
    );

    const handleMove = useCallback(
        (event: ReactPointerEvent<HTMLDivElement>) => {
            if (!dragging) return;
            const frame = frameAt(event.clientX);
            if (dragging === "first") {
                // Never past the anchor: see the note at the top of the file.
                props.onChange(Math.min(frame, props.anchor), props.last);
            } else {
                props.onChange(props.first, Math.max(frame, props.anchor));
            }
        },
        [dragging, frameAt, props],
    );

    const endDrag = useCallback(
        (event: ReactPointerEvent<HTMLDivElement>) => {
            if (!dragging) return;
            event.currentTarget.releasePointerCapture?.(event.pointerId);
            setDragging(null);
        },
        [dragging],
    );

    // What the run will produce, as opposed to what the handles span.
    const activeFirst =
        props.direction === "forward" ? props.anchor : props.first;
    const activeLast =
        props.direction === "backward" ? props.anchor : props.last;

    // Produced frames are reported as one band from the lowest to the highest:
    // propagation walks outward from the anchor, so the band is contiguous in
    // practice, and one overlay beats a DOM node per frame on a long clip.
    let producedFrom: number | null = null;
    let producedTo: number | null = null;
    if (props.produced && props.produced.size > 0) {
        producedFrom = Infinity;
        producedTo = -Infinity;
        for (const frame of props.produced) {
            if (frame < producedFrom) producedFrom = frame;
            if (frame > producedTo) producedTo = frame;
        }
    }

    return (
        <div
            className="propRange"
            style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                // Its own line under the other controls: a range is easier to
                // drag when it spans the whole panel.
                flex: "1 1 100%",
                minWidth: 260,
            }}
        >
            <div
                ref={trackRef}
                onPointerMove={handleMove}
                onPointerUp={endDrag}
                onPointerCancel={endDrag}
                style={{
                    position: "relative",
                    flex: 1,
                    height: 22,
                    borderRadius: 4,
                    background: "rgba(127, 127, 127, 0.18)",
                    border: "1px solid rgba(127, 127, 127, 0.35)",
                    cursor: "default",
                    touchAction: "none",
                }}
                title={`Frames 1–${props.frameCount}. Drag the handles to choose the range; the anchor frame must stay inside it.`}
            >
                {/* The range the handles span. */}
                <div
                    style={{
                        position: "absolute",
                        top: 0,
                        bottom: 0,
                        left: `${percent(props.first)}%`,
                        width: `${percent(props.last) - percent(props.first)}%`,
                        background: "rgba(127, 127, 127, 0.22)",
                    }}
                />
                {/* What the direction will actually produce. */}
                <div
                    style={{
                        position: "absolute",
                        top: 0,
                        bottom: 0,
                        left: `${percent(activeFirst)}%`,
                        width: `${percent(activeLast) - percent(activeFirst)}%`,
                        background: "rgba(46, 204, 113, 0.35)",
                    }}
                />
                {/* Masks produced by the running job. */}
                {producedFrom !== null && producedTo !== null && (
                    <div
                        style={{
                            position: "absolute",
                            top: 0,
                            bottom: 0,
                            left: `${percent(producedFrom)}%`,
                            width: `${percent(producedTo) - percent(producedFrom)}%`,
                            background: props.running
                                ? "rgba(255, 204, 51, 0.55)"
                                : "rgba(255, 204, 51, 0.3)",
                        }}
                    />
                )}
                {/* The anchor: the frame that carries the mask. */}
                <div
                    style={{
                        position: "absolute",
                        top: -3,
                        bottom: -3,
                        left: `calc(${percent(props.anchor)}% - 1px)`,
                        width: 2,
                        background: "#e74c3c",
                    }}
                    title={`Anchor: frame ${props.anchor + 1}`}
                />
                {(["first", "last"] as Handle[]).map((handle) => (
                    <div
                        key={handle}
                        role="slider"
                        aria-label={
                            handle === "first" ? "First frame" : "Last frame"
                        }
                        aria-valuemin={0}
                        aria-valuemax={lastFrame}
                        aria-valuenow={
                            handle === "first" ? props.first : props.last
                        }
                        tabIndex={0}
                        onPointerDown={(event) => {
                            event.preventDefault();
                            event.currentTarget.setPointerCapture?.(
                                event.pointerId,
                            );
                            setDragging(handle);
                        }}
                        onKeyDown={(event) => {
                            const step = event.shiftKey ? 10 : 1;
                            const delta =
                                event.key === "ArrowLeft"
                                    ? -step
                                    : event.key === "ArrowRight"
                                      ? step
                                      : 0;
                            if (!delta) return;
                            event.preventDefault();
                            if (handle === "first") {
                                props.onChange(
                                    Math.min(
                                        Math.max(0, props.first + delta),
                                        props.anchor,
                                    ),
                                    props.last,
                                );
                            } else {
                                props.onChange(
                                    props.first,
                                    Math.max(
                                        Math.min(lastFrame, props.last + delta),
                                        props.anchor,
                                    ),
                                );
                            }
                        }}
                        style={{
                            position: "absolute",
                            top: -2,
                            bottom: -2,
                            left: `calc(${percent(
                                handle === "first" ? props.first : props.last,
                            )}% - ${HANDLE_WIDTH / 2}px)`,
                            width: HANDLE_WIDTH,
                            borderRadius: 3,
                            background: "#4aa3ff",
                            cursor: "ew-resize",
                            boxShadow: "0 0 0 1px rgba(0,0,0,0.35)",
                        }}
                        title={`${handle === "first" ? "First" : "Last"} frame: ${
                            (handle === "first" ? props.first : props.last) + 1
                        }`}
                    />
                ))}
            </div>
            <span style={{ fontSize: 12, opacity: 0.8, whiteSpace: "nowrap" }}>
                {props.last - props.first} frame
                {props.last - props.first === 1 ? "" : "s"}
            </span>
        </div>
    );
}
