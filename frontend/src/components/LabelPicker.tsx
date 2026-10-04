import { useEffect, useRef } from "react";
import { createPortal } from "react-dom";
import type { Label } from "../types";
import { UNLABELLED_COLOR } from "../lib/palette";
import styles from "./LabelPicker.module.css";

const PANEL_WIDTH = 190;
const PANEL_HEIGHT = 120;

export interface LabelPickerProps {
    /** The button the panel is anchored to. */
    anchor: DOMRect;
    labels: Label[];
    /** The tracklet's current label, or `null` when unlabelled. */
    current: number | null;
    onPick: (labelId: number | null) => void;
    onNew: () => void;
    onClose: () => void;
}

/**
 * The label picker: every label as a colour block, its id drawn in the middle.
 *
 * A block is the whole choice — the number is what tells labels apart when the
 * palette has cycled, and red with a dash is "no label". Rendered in a portal
 * so the scrollable list it opens from cannot clip it.
 */
export function LabelPicker({
    anchor,
    labels,
    current,
    onPick,
    onNew,
    onClose,
}: LabelPickerProps) {
    const ref = useRef<HTMLDivElement | null>(null);

    useEffect(() => {
        const onKey = (event: KeyboardEvent) => {
            if (event.key === "Escape") onClose();
        };
        const onDown = (event: MouseEvent) => {
            if (ref.current && !ref.current.contains(event.target as Node)) {
                onClose();
            }
        };
        // The anchor moves when its list scrolls, so the panel closes rather
        // than drifting away from the row it belongs to.
        const onScroll = () => onClose();
        window.addEventListener("keydown", onKey);
        window.addEventListener("mousedown", onDown);
        window.addEventListener("scroll", onScroll, true);
        return () => {
            window.removeEventListener("keydown", onKey);
            window.removeEventListener("mousedown", onDown);
            window.removeEventListener("scroll", onScroll, true);
        };
    }, [onClose]);

    const left = Math.min(
        anchor.right + 8,
        window.innerWidth - PANEL_WIDTH - 8,
    );
    const top = Math.min(anchor.top, window.innerHeight - PANEL_HEIGHT - 8);
    const currentName =
        current === null
            ? "No label"
            : (labels.find((label) => label.id === current)?.name ??
              "No label");

    return createPortal(
        <div
            ref={ref}
            className={styles.panel}
            style={{ left, top, width: PANEL_WIDTH }}
            role="dialog"
            aria-label="Assign a label"
        >
            <div className={styles.grid}>
                <button
                    type="button"
                    className={`${styles.block} ${
                        current === null ? styles.active : ""
                    }`}
                    style={{ background: UNLABELLED_COLOR }}
                    onClick={() => onPick(null)}
                    title="No label"
                    aria-label="No label"
                >
                    –
                </button>
                {labels.map((label) => (
                    <button
                        key={label.id}
                        type="button"
                        className={`${styles.block} ${
                            current === label.id ? styles.active : ""
                        }`}
                        style={{ background: label.color }}
                        onClick={() => onPick(label.id)}
                        title={label.name}
                        aria-label={`${label.name}, label ${label.id}`}
                    >
                        {label.id}
                    </button>
                ))}
                <button
                    type="button"
                    className={styles.new}
                    onClick={onNew}
                    title="New label"
                    aria-label="New label"
                >
                    ＋
                </button>
            </div>
            <p className={styles.caption}>{currentName}</p>
        </div>,
        document.body,
    );
}
