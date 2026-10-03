import {
    useCallback,
    useEffect,
    useRef,
    useState,
    type CSSProperties,
    type RefObject,
} from "react";

/* ------------------------------------------------------------- panel layout
 *
 * The two draggable seams: the sidebar's left edge and the label list's top
 * edge. Sizes are remembered per browser under these keys, so a reviewer who
 * widens a panel keeps that width on the next project.
 */

const LAYOUT_KEYS = {
    panelW: "vsr.layout.panelW",
    labelsH: "vsr.layout.labelsH",
};

/** Mirrors `--rail-w`; the rail is fixed, so clamping the sidebar needs it. */
const RAIL_W = 72;
/** Neither the sidebar nor the stage may be squeezed below these. */
const PANEL_MIN_W = 280;
const STAGE_MIN_W = 360;
/** The label list keeps a header plus a couple of rows; the object list keeps more. */
const LABELS_MIN_H = 132;
const TRACKLETS_MIN_H = 168;
/** Mirrors the stylesheet's 30% fallback, for the first keyboard nudge. */
const LABELS_DEFAULT_FRAC = 0.3;

/** A remembered panel size, or `null` when there is nothing trustworthy. */
function readStoredSize(key: string): number | null {
    try {
        const raw = localStorage.getItem(key);
        const value = raw === null ? NaN : Number(raw);
        return Number.isFinite(value) && value > 0 ? value : null;
    } catch {
        return null;
    }
}

function storeSize(key: string, value: number): void {
    try {
        localStorage.setItem(key, String(value));
    } catch {
        /* storage is blocked: the size still holds for this page */
    }
}

function forgetSize(key: string): void {
    try {
        localStorage.removeItem(key);
    } catch {
        /* nothing to do */
    }
}

export interface PanelLayout {
    /** Attach to the body element the sidebar's seam measures against. */
    bodyRef: RefObject<HTMLDivElement | null>;
    /** Attach to the sidebar element the label seam measures against. */
    sidebarRef: RefObject<HTMLElement | null>;
    /** `--panel-w` override for the body, or `undefined` for the stylesheet default. */
    bodyStyle: CSSProperties | undefined;
    /** `--labels-h` override for the sidebar, or `undefined` for the default. */
    sidebarStyle: CSSProperties | undefined;
    onPanelDrag: (clientX: number) => void;
    onLabelsDrag: (clientY: number) => void;
    onPanelNudge: (delta: number) => void;
    onLabelsNudge: (delta: number) => void;
    resetPanelW: () => void;
    resetLabelsH: () => void;
}

/**
 * The workspace's two resizable seams.
 *
 * The hook owns the sizes, the clamping and the persistence; the caller only
 * attaches `bodyRef`/`sidebarRef` and forwards the pointer deltas from its
 * `Splitter`s. A size of `null` defers to the stylesheet — the `--panel-w`
 * token, the label list's content height — so an untouched workspace looks
 * exactly as it did before the seams existed; a number pins the panel in px.
 */
export function usePanelLayout(): PanelLayout {
    const [panelW, setPanelW] = useState<number | null>(() =>
        readStoredSize(LAYOUT_KEYS.panelW),
    );
    const [labelsH, setLabelsH] = useState<number | null>(() =>
        readStoredSize(LAYOUT_KEYS.labelsH),
    );
    const bodyRef = useRef<HTMLDivElement>(null);
    const sidebarRef = useRef<HTMLElement>(null);

    /** Pin the sidebar to `width`, clamped so the stage stays usable. */
    const applyPanelW = useCallback((width: number) => {
        const rect = bodyRef.current?.getBoundingClientRect();
        if (!rect) return;
        const max = Math.max(PANEL_MIN_W, rect.width - RAIL_W - STAGE_MIN_W);
        const next = Math.round(Math.min(Math.max(width, PANEL_MIN_W), max));
        setPanelW(next);
        storeSize(LAYOUT_KEYS.panelW, next);
    }, []);

    /** Pin the label list to `height`, clamped so the object list stays usable. */
    const applyLabelsH = useCallback((height: number) => {
        const rect = sidebarRef.current?.getBoundingClientRect();
        if (!rect || rect.height === 0) return;
        const max = Math.max(LABELS_MIN_H, rect.height - TRACKLETS_MIN_H);
        const next = Math.round(Math.min(Math.max(height, LABELS_MIN_H), max));
        setLabelsH(next);
        storeSize(LAYOUT_KEYS.labelsH, next);
    }, []);

    /*
     * A drag reports the pointer, not the size: the sidebar runs from the
     * pointer to the body's right edge, the label list from the pointer down.
     */
    const onPanelDrag = useCallback(
        (clientX: number) => {
            const rect = bodyRef.current?.getBoundingClientRect();
            if (rect) applyPanelW(rect.right - clientX);
        },
        [applyPanelW],
    );

    const onLabelsDrag = useCallback(
        (clientY: number) => {
            const rect = sidebarRef.current?.getBoundingClientRect();
            if (rect) applyLabelsH(rect.bottom - clientY);
        },
        [applyLabelsH],
    );

    /*
     * Arrow keys step from the size actually in force, which before the first
     * drag is a measured width / the stylesheet's 30% — not a stored number.
     */
    const onPanelNudge = useCallback(
        (delta: number) => {
            const width = sidebarRef.current?.getBoundingClientRect().width;
            if (width) applyPanelW(width - delta);
        },
        [applyPanelW],
    );

    const onLabelsNudge = useCallback(
        (delta: number) => {
            const rect = sidebarRef.current?.getBoundingClientRect();
            if (!rect || rect.height === 0) return;
            const current = labelsH ?? rect.height * LABELS_DEFAULT_FRAC;
            applyLabelsH(current - delta);
        },
        [applyLabelsH, labelsH],
    );

    const resetPanelW = useCallback(() => {
        setPanelW(null);
        forgetSize(LAYOUT_KEYS.panelW);
    }, []);

    const resetLabelsH = useCallback(() => {
        setLabelsH(null);
        forgetSize(LAYOUT_KEYS.labelsH);
    }, []);

    // A size set at a larger window can squeeze the stage or the object list out
    // of a smaller one; re-clamp on resize (and once on mount, for a stored size).
    useEffect(() => {
        const onResize = () => {
            if (panelW !== null) applyPanelW(panelW);
            if (labelsH !== null) applyLabelsH(labelsH);
        };
        window.addEventListener("resize", onResize);
        return () => window.removeEventListener("resize", onResize);
    }, [panelW, labelsH, applyPanelW, applyLabelsH]);

    const bodyStyle =
        panelW === null
            ? undefined
            : ({ "--panel-w": `${panelW}px` } as CSSProperties);
    const sidebarStyle =
        labelsH === null
            ? undefined
            : ({ "--labels-h": `${labelsH}px` } as CSSProperties);

    return {
        bodyRef,
        sidebarRef,
        bodyStyle,
        sidebarStyle,
        onPanelDrag,
        onLabelsDrag,
        onPanelNudge,
        onLabelsNudge,
        resetPanelW,
        resetLabelsH,
    };
}
