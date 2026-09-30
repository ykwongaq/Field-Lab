import { useEffect, useRef, useState } from "react";
import type { Sam3Status } from "../lib/sam3Api";
import type { ProjectMode } from "../lib/project";
import { MODE_VOCABULARY } from "../lib/project";
import { Icon, type IconName } from "../ui";
import styles from "./Toolbar.module.css";

export type Tool = "review" | "addMask" | "editMask" | "propagate";

export type DrawMethod = "point" | "box" | "text" | "polygon" | "brush";

export type DeleteScope = "frame" | "tracklet";

interface ToolbarProps {
    mode: ProjectMode;
    tool: Tool;
    sam: Sam3Status | null;
    modelName: string;
    canDeleteFrame: boolean;
    canEdit: boolean;
    canDeleteTracklet: boolean;
    canPropagate: boolean;
    propagateModel: string;
    selectedMaskCount: number;
    onToolChange: (tool: Tool) => void;
    onDelete: (scope: DeleteScope) => void;
    onRefreshStatus: () => void;
}

/**
 * The tool rail.
 *
 * One column, one tool at a time, with the shortcut printed under each label.
 * The footer reports the segmentation model so its availability is visible
 * without opening anything.
 */
export function Toolbar(props: ToolbarProps) {
    const vocab = MODE_VOCABULARY[props.mode];
    const [deleteOpen, setDeleteOpen] = useState(false);
    const deleteRef = useRef<HTMLDivElement>(null);

    // Close the delete menu on outside click / Escape.
    useEffect(() => {
        if (!deleteOpen) return;
        const onPointerDown = (event: MouseEvent) => {
            if (!deleteRef.current?.contains(event.target as Node))
                setDeleteOpen(false);
        };
        const onKey = (event: KeyboardEvent) => {
            if (event.key === "Escape") setDeleteOpen(false);
        };
        document.addEventListener("mousedown", onPointerDown);
        document.addEventListener("keydown", onKey);
        return () => {
            document.removeEventListener("mousedown", onPointerDown);
            document.removeEventListener("keydown", onKey);
        };
    }, [deleteOpen]);

    const model = props.modelName;
    const samReady = props.sam?.available ?? false;

    const tool = (
        value: Tool,
        label: string,
        icon: IconName,
        shortcut: string | undefined,
        enabled: boolean,
        title: string,
    ) => (
        <ToolButton
            label={label}
            icon={icon}
            shortcut={shortcut}
            active={props.tool === value}
            disabled={!enabled}
            title={title}
            onClick={() => props.onToolChange(value)}
        />
    );

    const statusLabel = !props.sam
        ? `Checking ${model}…`
        : samReady
          ? `${model} ready${props.sam.loaded ? "" : " (loads on first use)"}`
          : `${model} unavailable${props.sam.error ? ` — ${props.sam.error}` : ""}`;

    return (
        <nav className={styles.rail} aria-label="Tools">
            {tool(
                "review",
                "Select",
                "select",
                "Esc",
                true,
                `Browse and inspect masks without editing (Esc)`,
            )}
            {tool(
                "addMask",
                "Add",
                "plus",
                "A",
                true,
                `Draw a new ${vocab.unit} on this frame (A)`,
            )}
            {tool(
                "editMask",
                "Edit",
                "pencil",
                "E",
                props.canEdit,
                props.canEdit
                    ? `Correct the selected ${vocab.unit}'s mask on this frame (E)`
                    : `Select a ${vocab.unit} first`,
            )}

            <div ref={deleteRef} className={styles.menuAnchor}>
                <ToolButton
                    label="Delete"
                    icon="trash"
                    active={deleteOpen}
                    disabled={!props.canDeleteTracklet}
                    title={
                        props.canDeleteTracklet
                            ? `Remove a mask or the whole ${vocab.unit}`
                            : `Select a ${vocab.unit} first`
                    }
                    onClick={() => setDeleteOpen((open) => !open)}
                />
                {deleteOpen && (
                    <div className={styles.menu} role="menu">
                        <button
                            type="button"
                            role="menuitem"
                            className={styles.menuItem}
                            disabled={!props.canDeleteFrame}
                            onClick={() => {
                                setDeleteOpen(false);
                                props.onDelete("frame");
                            }}
                        >
                            <span>This frame only</span>
                            <span className={styles.menuHint}>
                                {props.canDeleteFrame
                                    ? props.selectedMaskCount <= 1
                                        ? `Last mask — removes the ${vocab.unit}`
                                        : `${props.selectedMaskCount - 1} frames keep their mask`
                                    : "No mask on this frame"}
                            </span>
                        </button>
                        <button
                            type="button"
                            role="menuitem"
                            className={`${styles.menuItem} ${styles.menuDanger}`}
                            onClick={() => {
                                setDeleteOpen(false);
                                props.onDelete("tracklet");
                            }}
                        >
                            <span>Whole {vocab.unit}</span>
                            <span className={styles.menuHint}>
                                All {props.selectedMaskCount} mask
                                {props.selectedMaskCount === 1 ? "" : "s"}
                            </span>
                        </button>
                    </div>
                )}
            </div>

            <div className={styles.divider} />

            {tool(
                "propagate",
                "Track",
                "propagate",
                "T",
                props.canPropagate || props.tool === "propagate",
                props.canPropagate
                    ? `Carry this mask across frames with ${props.propagateModel} (T)`
                    : `Select a ${vocab.unit} with a mask on this frame first`,
            )}

            <div className={styles.spacer} />

            <button
                type="button"
                className={styles.status}
                onClick={props.onRefreshStatus}
                title={`${statusLabel}. Click to re-check.`}
                aria-label={statusLabel}
            >
                <span
                    className={`${styles.dot} ${
                        !props.sam
                            ? styles.dotPending
                            : samReady
                              ? styles.dotOk
                              : styles.dotBad
                    }`}
                />
                <span className={styles.statusText}>
                    {props.sam ? (samReady ? "Model" : "Offline") : "…"}
                </span>
            </button>
        </nav>
    );
}

interface ToolButtonProps {
    label: string;
    icon: IconName;
    title: string;
    onClick: () => void;
    active?: boolean;
    disabled?: boolean;
    shortcut?: string;
}

function ToolButton(props: ToolButtonProps) {
    return (
        <button
            type="button"
            className={`${styles.tool} ${props.active ? styles.toolActive : ""}`}
            disabled={props.disabled}
            aria-pressed={props.active}
            title={
                props.shortcut
                    ? `${props.title} · ${props.shortcut}`
                    : props.title
            }
            onClick={props.onClick}
        >
            <Icon name={props.icon} size={19} />
            <span className={styles.label}>{props.label}</span>
        </button>
    );
}
