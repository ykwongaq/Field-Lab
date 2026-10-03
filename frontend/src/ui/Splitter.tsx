import { useCallback, useState } from "react";
import type {
    KeyboardEvent as ReactKeyboardEvent,
    PointerEvent as ReactPointerEvent,
} from "react";
import styles from "./Splitter.module.css";

export interface SplitterProps {
    /**
     * `"vertical"` is a full-height seam dragged left/right to resize the
     * columns on either side; `"horizontal"` is a full-width seam dragged
     * up/down.
     */
    orientation: "vertical" | "horizontal";
    /**
     * A drag in progress, reported as the pointer's position on the seam's own
     * axis (`clientX` when vertical, `clientY` when horizontal). Clamping stays
     * with the caller, which is the only party that knows the layout.
     */
    onDrag: (position: number) => void;
    /** An arrow key press, as a signed step along the seam's axis. */
    onNudge?: (delta: number) => void;
    /** Double-click (or Enter) puts the panel back to its default size. */
    onReset?: () => void;
    /** Describes what this seam resizes, for assistive tech. */
    label: string;
}

/**
 * The draggable edge between two panels.
 *
 * It owns the pointer bookkeeping — capture, so a fast drag that outruns the
 * handle keeps resizing, then release — and leaves the arithmetic to the
 * caller. That split keeps the component ignorant of the layout it divides, so
 * the same part serves both the sidebar's left edge and the label list's top.
 */
export function Splitter({
    orientation,
    onDrag,
    onNudge,
    onReset,
    label,
}: SplitterProps) {
    const [dragging, setDragging] = useState(false);

    const handlePointerDown = useCallback(
        (event: ReactPointerEvent<HTMLDivElement>) => {
            if (event.button !== 0) return;
            // Focus explicitly: the default is unreliable here, and a keyboard
            // user who grabs the seam should be able to nudge it afterwards.
            event.preventDefault();
            event.currentTarget.focus();
            event.currentTarget.setPointerCapture(event.pointerId);
            setDragging(true);
        },
        [],
    );

    const handlePointerMove = useCallback(
        (event: ReactPointerEvent<HTMLDivElement>) => {
            if (!event.currentTarget.hasPointerCapture(event.pointerId)) return;
            onDrag(orientation === "vertical" ? event.clientX : event.clientY);
        },
        [onDrag, orientation],
    );

    const endDrag = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
        if (event.currentTarget.hasPointerCapture(event.pointerId)) {
            event.currentTarget.releasePointerCapture(event.pointerId);
        }
        setDragging(false);
    }, []);

    const handleKeyDown = useCallback(
        (event: ReactKeyboardEvent<HTMLDivElement>) => {
            /*
             * Stop propagation, not just the default: the workspace listens on
             * `window` for the same keys (arrows step frames, Enter commits a
             * mask), so nudging a seam would otherwise also move the clip on.
             */
            if (event.key === "Enter" && onReset) {
                event.preventDefault();
                event.stopPropagation();
                onReset();
                return;
            }
            if (!onNudge) return;
            // A vertical seam divides columns, so it answers to Left/Right.
            const back = orientation === "vertical" ? "ArrowLeft" : "ArrowUp";
            const forward =
                orientation === "vertical" ? "ArrowRight" : "ArrowDown";
            const step = event.shiftKey ? 48 : 16;
            if (event.key === back) {
                event.preventDefault();
                event.stopPropagation();
                onNudge(-step);
            } else if (event.key === forward) {
                event.preventDefault();
                event.stopPropagation();
                onNudge(step);
            }
        },
        [onNudge, onReset, orientation],
    );

    return (
        <div
            role="separator"
            aria-orientation={orientation}
            aria-label={label}
            title={`${label} — drag, or use the arrow keys`}
            tabIndex={0}
            className={`${styles.splitter} ${
                orientation === "vertical" ? styles.vertical : styles.horizontal
            } ${dragging ? styles.dragging : ""}`}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={endDrag}
            onPointerCancel={endDrag}
            onKeyDown={handleKeyDown}
            onDoubleClick={onReset}
        />
    );
}
