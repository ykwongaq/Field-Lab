import { useEffect, type Dispatch, type SetStateAction } from "react";
import type { DrawMethod, Tool } from "../components/Toolbar";

/** The workspace state the key map reads, flattened to the values it needs. */
export interface WorkspaceShortcutState {
    /** While the export chooser is up, no shortcut may reach the workspace behind. */
    exportOpen: boolean;
    tool: Tool;
    method: DrawMethod;
    /** Polygon vertices drawn so far; Enter closes it at three or more. */
    polygonLength: number;
    /** Whether an object is selected, which gates the Edit-mask shortcut. */
    hasSelection: boolean;
    /** Whether the selected object has a mask a human drew here, which gates Propagate. */
    selectedHasHumanMaskHere: boolean;
    /** Whether a propagation result is under review, which makes Enter accept it. */
    hasPropRun: boolean;
    /** Whether a run is still producing frames, which locks the keyboard. */
    propagating: boolean;
}

/** The actions the key map can trigger. */
export interface WorkspaceShortcutHandlers {
    stepFrame: (delta: number) => void;
    changeTool: (tool: Tool) => void;
    changeMethod: (method: DrawMethod) => void;
    closePolygon: () => void;
    commitMask: () => void;
    undo: () => void;
    acceptPropagation: () => void;
    runPropagation: () => Promise<void>;
    setPlaying: Dispatch<SetStateAction<boolean>>;
    setBrushSize: Dispatch<SetStateAction<number>>;
}

/**
 * The workspace keyboard map.
 *
 * Typing into a form control is left alone, and the export chooser is modal: while
 * it is up its own Escape handler closes it and no shortcut may reach the workspace
 * behind. Otherwise the keys are layered by tool — the drawing tools own Enter,
 * Escape, Backspace and the method letters, and the always-on keys sit underneath.
 *
 * State is passed flattened to the values the map actually branches on, so a key
 * map that only cares whether something is selected does not re-subscribe when the
 * selection changes from one object to another.
 */
export function useWorkspaceShortcuts(
    state: WorkspaceShortcutState,
    handlers: WorkspaceShortcutHandlers,
): void {
    const {
        exportOpen,
        tool,
        method,
        polygonLength,
        hasSelection,
        selectedHasHumanMaskHere,
        hasPropRun,
        propagating,
    } = state;
    const {
        stepFrame,
        changeTool,
        changeMethod,
        closePolygon,
        commitMask,
        undo,
        acceptPropagation,
        runPropagation,
        setPlaying,
        setBrushSize,
    } = handlers;

    useEffect(() => {
        const onKeyDown = (event: KeyboardEvent) => {
            const target = event.target as HTMLElement | null;
            if (
                target &&
                ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)
            )
                return;

            // The export chooser is modal: while it is up, its own Escape
            // handler closes it and no shortcut may reach the workspace behind.
            if (exportOpen) return;

            // A live run owns the keyboard: it drives the frame itself and its only
            // exit is the bar's Stop button, so no shortcut may switch tools.
            if (propagating) return;

            if (tool === "propagate") {
                switch (event.key) {
                    case "Escape":
                        event.preventDefault();
                        changeTool("review");
                        return;
                    case "Enter":
                        event.preventDefault();
                        if (hasPropRun) acceptPropagation();
                        else void runPropagation();
                        return;
                }
            } else if (tool !== "review") {
                switch (event.key) {
                    case "Escape":
                        event.preventDefault();
                        changeTool("review");
                        return;
                    case "Enter":
                        event.preventDefault();
                        if (method === "polygon" && polygonLength >= 3)
                            closePolygon();
                        else commitMask();
                        return;
                    case "Backspace":
                        event.preventDefault();
                        undo();
                        return;
                    case "s":
                        changeMethod("point");
                        return;
                    case "p":
                        changeMethod("polygon");
                        return;
                    case "b":
                        changeMethod("brush");
                        return;
                    case "[":
                        setBrushSize((size) =>
                            Math.max(1, Math.round(size / 1.25)),
                        );
                        return;
                    case "]":
                        setBrushSize((size) =>
                            Math.min(200, Math.round(size * 1.25)),
                        );
                        return;
                }
            }

            switch (event.key) {
                case "a":
                    changeTool(tool === "addMask" ? "review" : "addMask");
                    break;
                case "e":
                    if (hasSelection)
                        changeTool(tool === "editMask" ? "review" : "editMask");
                    break;
                case "t":
                    if (selectedHasHumanMaskHere || tool === "propagate")
                        changeTool(
                            tool === "propagate" ? "review" : "propagate",
                        );
                    break;
                case " ":
                    event.preventDefault();
                    setPlaying((value) => !value);
                    break;
                case "ArrowLeft":
                    event.preventDefault();
                    stepFrame(-1);
                    break;
                case "ArrowRight":
                    event.preventDefault();
                    stepFrame(1);
                    break;
            }
        };
        window.addEventListener("keydown", onKeyDown);
        return () => window.removeEventListener("keydown", onKeyDown);
    }, [
        stepFrame,
        tool,
        method,
        polygonLength,
        hasSelection,
        changeTool,
        changeMethod,
        closePolygon,
        commitMask,
        undo,
        selectedHasHumanMaskHere,
        hasPropRun,
        propagating,
        acceptPropagation,
        runPropagation,
        exportOpen,
        setPlaying,
        setBrushSize,
    ]);
}
