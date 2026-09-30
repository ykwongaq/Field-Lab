/**
 * The frame canvas view transform: screen <-> frame coordinates.
 *
 * The panel owns one `View` (a zoom multiplier on top of the uniform
 * fit-to-container scale, plus a pan offset in screen pixels) and derives
 * everything else from it. `frameLayout()` is the single place that turns a view
 * into the `{ x, y, scale }` mapping the rest of the panel already works in, so
 * zoom and pan need no changes to prompt placement, hit-testing, brush strokes
 * or the vector overlay.
 *
 * Pure functions, no DOM: the arithmetic here is the part worth checking, and
 * this way it can be run in Node.
 */

export interface Size {
    w: number;
    h: number;
}

export interface View {
    /** Multiplier on the fit scale. 1 shows the whole frame. */
    zoom: number;
    /** Screen-pixel offset from the centred fit position. */
    panX: number;
    panY: number;
}

/** Screen-space placement of the frame inside the panel, plus its frame size. */
export interface FrameLayout {
    x: number;
    y: number;
    scale: number;
    width: number;
    height: number;
}

export const MIN_ZOOM = 1;
export const MAX_ZOOM = 16;

/** Fit: whole frame visible, centred, unpanned. */
export const FIT_VIEW: View = { zoom: 1, panX: 0, panY: 0 };

/** The uniform scale at which the frame just fits the container. */
export function fitScale(container: Size, frame: Size): number {
    if (container.w <= 0 || container.h <= 0) return 0;
    if (frame.w <= 0 || frame.h <= 0) return 0;
    return Math.min(container.w / frame.w, container.h / frame.h);
}

export function clampZoom(zoom: number): number {
    if (!Number.isFinite(zoom)) return MIN_ZOOM;
    return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, zoom));
}

/** Where the frame sits, in CSS pixels, for a given view. */
export function frameLayout(
    container: Size,
    frame: Size,
    view: View,
): FrameLayout {
    const scale = fitScale(container, frame) * view.zoom;
    const drawWidth = frame.w * scale;
    const drawHeight = frame.h * scale;
    return {
        x: (container.w - drawWidth) / 2 + view.panX,
        y: (container.h - drawHeight) / 2 + view.panY,
        scale,
        width: frame.w,
        height: frame.h,
    };
}

/**
 * How far the frame may slide along one axis. A frame smaller than the
 * container is locked centred; a larger one keeps covering the container, so it
 * can never be dragged off into empty space.
 */
function panLimits(container: number, scaled: number): [number, number] {
    const slack = (container - scaled) / 2;
    return slack >= 0 ? [0, 0] : [slack, -slack];
}

/** Clamp a view's pan so the frame stays sensibly placed. */
export function clampPan(container: Size, frame: Size, view: View): View {
    const scale = fitScale(container, frame) * view.zoom;
    const [minX, maxX] = panLimits(container.w, frame.w * scale);
    const [minY, maxY] = panLimits(container.h, frame.h * scale);
    const panX = Math.min(maxX, Math.max(minX, view.panX));
    const panY = Math.min(maxY, Math.max(minY, view.panY));
    if (panX === view.panX && panY === view.panY) return view;
    return { ...view, panX, panY };
}

/**
 * Zoom to `zoom` while keeping the frame point under `cursor` (container CSS
 * pixels) exactly where it is — the difference between a viewer that feels like
 * a magnifier and one that lurches away from what you were looking at.
 */
export function zoomAbout(
    container: Size,
    frame: Size,
    view: View,
    zoom: number,
    cursor: { x: number; y: number },
): View {
    const next = clampZoom(zoom);
    if (next === view.zoom) return view;
    const before = frameLayout(container, frame, view);
    // The centred position at the new zoom (panX/panY = 0), which is what the
    // fresh pan offset has to be measured against.
    const centred = frameLayout(container, frame, {
        zoom: next,
        panX: 0,
        panY: 0,
    });
    const k = next / view.zoom;
    const zoomed: View = {
        zoom: next,
        panX: cursor.x - (cursor.x - before.x) * k - centred.x,
        panY: cursor.y - (cursor.y - before.y) * k - centred.y,
    };
    return clampPan(container, frame, zoomed);
}

export function panBy(
    container: Size,
    frame: Size,
    view: View,
    dx: number,
    dy: number,
): View {
    return clampPan(container, frame, {
        ...view,
        panX: view.panX + dx,
        panY: view.panY + dy,
    });
}

/**
 * Wheel delta -> zoom multiplier. Wheel deltas arrive in three units depending
 * on the device and the browser's scroll setting, so normalise to pixels first,
 * then map exponentially: the same amount of scrolling always changes the zoom
 * by the same ratio, at any zoom level.
 */
export function wheelZoomFactor(deltaY: number, deltaMode: number): number {
    const pixels =
        deltaMode === 1
            ? deltaY * 16 // lines
            : deltaMode === 2
              ? deltaY * 100 // pages
              : deltaY;
    return Math.exp(-pixels * 0.0015);
}

/** Container CSS pixels -> frame pixels. */
export function screenToFrame(
    layout: FrameLayout,
    x: number,
    y: number,
): { x: number; y: number } {
    return {
        x: (x - layout.x) / layout.scale,
        y: (y - layout.y) / layout.scale,
    };
}
