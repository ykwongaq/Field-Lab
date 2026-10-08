import { useCallback, useEffect, useRef, useState } from "react";
import type {
    MouseEvent as ReactMouseEvent,
    PointerEvent as ReactPointerEvent,
} from "react";
import type { Clip } from "../lib/clip";
import type { FrameSource } from "../lib/frames";
import { FrameCache } from "../lib/frameCache";
import { MaskRenderer, boundaryWidthFor, ringsToPath } from "../lib/mask";
import { MaskCache, type MaskRequest } from "../lib/maskApi";
import { formatTimecode } from "../lib/format";
import { SELECTED_COLOR } from "../lib/palette";
import { Icon } from "../ui";
import {
    RasterCanvas,
    runsArea,
    runsContain,
    type FramePoint,
    type PaintMode,
} from "../lib/raster";
import {
    insertVertex,
    moveVertex,
    nearestEdge,
    nearestVertex,
    outlineFromRuns,
    removeVertex,
} from "../lib/contour";
import type { DecodedMask, PromptPoint, RawRle, Tracklet } from "../types";
import type { PromptBox } from "../lib/sam3Api";
import {
    FIT_VIEW,
    MAX_ZOOM,
    MIN_ZOOM,
    clampZoom,
    frameLayout,
    panBy,
    screenToFrame,
    visibleSlice,
    wheelZoomFactor,
    zoomAbout,
    type FrameLayout,
    type Size,
    type View,
} from "../lib/viewTransform";
import type { DrawMethod, Tool } from "./Toolbar";
import styles from "./VideoPanel.module.css";

export const PREVIEW_COLOR = "#ffcc33";
const CANDIDATE_ADD_COLOR = "#ff8c42";
const CANDIDATE_ERASE_COLOR = "#ff5f5f";
const POSITIVE_COLOR = "#2ecc71";
const NEGATIVE_COLOR = "#e74c3c";
const CLOSE_POLYGON_RADIUS = 9;
/** Screen-space radii for grabbing a vertex or an edge of an existing outline. */
const VERTEX_HIT_RADIUS = 7;
const EDGE_HIT_RADIUS = 6;
/** One marching-ants dash cycle, in milliseconds. */
const ANTS_PERIOD = 600;

interface VideoPanelProps {
    clip: Clip;
    frames: FrameSource;
    frameIndex: number;
    playing: boolean;
    selectedTrackletId: number | null;
    maskOpacity: number;
    onFrameChange: (frame: number) => void;
    onPlayToggle: () => void;
    onStep: (delta: number) => void;
    onMaskOpacityChange: (value: number) => void;
    tool: Tool;
    method: DrawMethod;
    paintMode: PaintMode;
    brushSize: number;
    prompt: PromptPoint[];
    polygon: FramePoint[];
    /**
     * Editable outline of the mask being corrected, derived from the draft.
     *
     * Non-null means the polygon tool is in *outline edit* mode: the shape is
     * already closed and its vertices are dragged, rather than clicked into place.
     */
    outline: FramePoint[][] | null;
    draft: DecodedMask | null;
    candidate: DecodedMask | null;
    /**
     * Colour for the candidate overlay. Unset means the prompt colours (orange to
     * add, red to erase); a propagation passes the object's own label colour, so a
     * preview is drawn on the video exactly like a committed mask and only the
     * timeline tells the two apart.
     */
    candidateColor?: string;
    editingTrackletId: number | null;
    onPromptPoint: (point: PromptPoint) => void;
    /** A box drag finished; the reviewer segments the object inside it. */
    onPromptBox?: (rect: PromptBox) => void;
    /** A box that is part of the current prompt (shown so it can be cleared). */
    box?: PromptBox | null;
    onPolygonPoint: (point: FramePoint) => void;
    onPolygonClose: () => void;
    /** The outline was edited: the rings to bake into the draft. */
    onOutlineChange: (rings: FramePoint[][]) => void;
    onStroke: (stroke: RawRle, mode: PaintMode) => void;
    onSelectTracklet: (id: number) => void;
    promptHint?: string;
}

export function VideoPanel(props: VideoPanelProps) {
    const wrapRef = useRef<HTMLDivElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    //: The marching-ants layer, above the frame canvas and repainted on its own.
    const antsRef = useRef<HTMLCanvasElement>(null);
    //: A pending "layout has not settled" retry, so it can be cancelled with the
    //: draw it belongs to.
    const retryRef = useRef<number | null>(null);

    const cacheRef = useRef<FrameCache | null>(null);
    if (!cacheRef.current) {
        cacheRef.current = new FrameCache(props.frames);
    }
    const maskRef = useRef<MaskRenderer | null>(null);
    // Rebuild whenever the frame size changes. The renderer is an offscreen
    // canvas of the clip's size, and compositing a 0x0 canvas throws — which used
    // to kill the whole paint, leaving a blank panel with no error on screen. A
    // renderer built before the size was known must therefore be replaced, and if
    // the size is still unknown it gets a nominal one so it is never null and the
    // frame keeps painting.
    const maskWidth = props.clip.width > 0 ? props.clip.width : 1920;
    const maskHeight = props.clip.height > 0 ? props.clip.height : 1080;
    if (
        !maskRef.current ||
        maskRef.current.canvasElement.width !== maskWidth ||
        maskRef.current.canvasElement.height !== maskHeight
    ) {
        maskRef.current = new MaskRenderer(maskWidth, maskHeight);
    }
    // Boundaries get their own buffer: they are blitted fully opaque, while the
    // fill beneath them follows the overlay-opacity slider.
    const edgeRef = useRef<MaskRenderer | null>(null);
    if (
        !edgeRef.current ||
        edgeRef.current.canvasElement.width !== maskWidth ||
        edgeRef.current.canvasElement.height !== maskHeight
    ) {
        edgeRef.current = new MaskRenderer(maskWidth, maskHeight);
    }
    // The selected object's outline, kept between animation frames (and across
    // pans and zooms): tracing it is the only costly part of the effect, and it
    // only changes when the clip, the object or the frame does.
    const antsPathRef = useRef<Path2D | null>(null);
    const antsKeyRef = useRef<{
        clip: Clip;
        trackletId: number;
        frameIndex: number;
    } | null>(null);
    // Where the dashes have marched to, kept across replans so a pan or zoom does
    // not snap them back to the start.
    const antsPhaseRef = useRef(0);
    const maskCacheRef = useRef<MaskCache | null>(null);
    if (!maskCacheRef.current) {
        maskCacheRef.current = new MaskCache();
    }

    const paintedFrameRef = useRef(-1);

    const layoutRef = useRef<FrameLayout | null>(null);
    const [layout, setLayout] = useState<FrameLayout | null>(null);

    // Zoom + pan. The ref is what the pointer and wheel handlers read, so a fast
    // drag never works from a layout a render behind; the state is what makes the
    // paint effect run.
    const viewRef = useRef<View>(FIT_VIEW);
    const [view, setView] = useState<View>(FIT_VIEW);
    const applyView = useCallback((next: View) => {
        viewRef.current = next;
        setView((current) =>
            current.zoom === next.zoom &&
            current.panX === next.panX &&
            current.panY === next.panY
                ? current
                : next,
        );
    }, []);
    const [panning, setPanning] = useState(false);
    const panRef = useRef<{
        pointerId: number;
        button: number;
        lastX: number;
        lastY: number;
        moved: boolean;
    } | null>(null);
    //: A right-drag that panned must not also land as a right-click.
    const swallowRightClickRef = useRef(false);

    const rasterRef = useRef<RasterCanvas | null>(null);
    if (!rasterRef.current) {
        rasterRef.current = new RasterCanvas(
            props.clip.width,
            props.clip.height,
        );
    }
    const strokeRef = useRef<{
        mode: PaintMode;
        last: FramePoint;
        pointerId: number;
    } | null>(null);
    //: The live outline drag. The rings live on the ref as well as in state so a
    //: fast drag never works from the previous render's snapshot.
    const dragRef = useRef<{
        pointerId: number;
        ring: number;
        index: number;
        rings: FramePoint[][];
    } | null>(null);
    const [dragRings, setDragRings] = useState<FramePoint[][] | null>(null);

    const [cursor, setCursor] = useState<FramePoint | null>(null);
    //: The box being dragged (box prompt): live while dragging, committed on release.
    const [boxRect, setBoxRect] = useState<PromptBox | null>(null);
    const boxStartRef = useRef<FramePoint | null>(null);

    /**
     * The objects whose masks the canvas draws, by tool.
     *
     * - **Select** shows every object: seeing the neighbours is how you tell
     *   whether two animals' masks are being confused.
     * - **Add** shows every object too, so a new mask is drawn against the
     *   existing ones rather than a bare frame. The live draft and the model's
     *   candidate are layered on top separately.
     * - **Edit** shows only the object being corrected — the neighbours are noise
     *   when the job is to fix one mask. Its own committed mask is held back too
     *   (the loop below skips `editingTrackletId`), so what you see is the draft.
     * - **Track** shows only the object being carried across frames, so the run
     *   under review is not lost among the rest.
     *
     * In Select the selected object is painted in `SELECTED_COLOR` on top of the
     * rest (see the paint effect); every other mask keeps its tracklet colour.
     */
    const maskTracklets = useCallback((): Tracklet[] => {
        switch (props.tool) {
            case "review":
            case "addMask":
                return props.clip.tracklets;
            case "editMask":
            case "propagate": {
                const focus =
                    props.editingTrackletId ?? props.selectedTrackletId;
                return props.clip.tracklets.filter((t) => t.id === focus);
            }
        }
    }, [
        props.tool,
        props.clip.tracklets,
        props.editingTrackletId,
        props.selectedTrackletId,
    ]);

    const drawing = props.tool === "addMask" || props.tool === "editMask";

    // Where the right button already means something on the canvas: a negative
    // point, closing the polygon, erasing a brush stroke. Panning must not steal
    // those, so right-drag only pans where the right button is otherwise free.
    const rightHasDrawMeaning =
        drawing &&
        (props.method === "point" ||
            props.method === "polygon" ||
            props.method === "brush");

    const containerSize = useCallback((): Size => {
        const bounds = wrapRef.current?.getBoundingClientRect();
        return { w: bounds?.width ?? 0, h: bounds?.height ?? 0 };
    }, []);

    const frameSize = useCallback((): Size => {
        const layout = layoutRef.current;
        return layout
            ? { w: layout.width, h: layout.height }
            : { w: props.clip.width, h: props.clip.height };
    }, [props.clip.width, props.clip.height]);

    const zoomBy = useCallback(
        (factor: number) => {
            const container = containerSize();
            const layout = layoutRef.current;
            if (!layout) return;
            applyView(
                zoomAbout(
                    container,
                    frameSize(),
                    viewRef.current,
                    clampZoom(viewRef.current.zoom * factor),
                    { x: container.w / 2, y: container.h / 2 },
                ),
            );
        },
        [applyView, containerSize, frameSize],
    );

    const resetView = useCallback(() => applyView(FIT_VIEW), [applyView]);

    // A new clip is a new frame size: start from fit rather than inheriting the
    // previous clip's zoom and pan.
    useEffect(() => {
        applyView(FIT_VIEW);
    }, [props.clip, applyView]);

    const toFrame = useCallback(
        (
            event: { clientX: number; clientY: number },
            clamp = false,
        ): FramePoint | null => {
            const layout = layoutRef.current;
            const canvas = canvasRef.current;
            if (!layout || !canvas) return null;
            const rect = canvas.getBoundingClientRect();
            const { x, y } = screenToFrame(
                layout,
                event.clientX - rect.left,
                event.clientY - rect.top,
            );
            if (x < 0 || y < 0 || x >= layout.width || y >= layout.height) {
                if (!clamp) return null;
                return {
                    x: Math.min(layout.width, Math.max(0, x)),
                    y: Math.min(layout.height, Math.max(0, y)),
                };
            }
            return { x, y };
        },
        [],
    );

    const selectAt = useCallback(
        async (point: FramePoint) => {
            const cache = maskCacheRef.current;
            if (!cache) return;
            const frameIndex = props.frameIndex;
            const requests: MaskRequest[] = [];
            const owners: Tracklet[] = [];
            for (const tracklet of props.clip.tracklets) {
                const payload = props.clip.rawMaskAt(tracklet, frameIndex);
                if (!payload) continue;
                requests.push({ trackletId: tracklet.id, frameIndex, payload });
                owners.push(tracklet);
            }
            if (requests.length === 0) return;
            const decoded = await cache.resolveBatch(requests);
            let bestId: number | null = null;
            let bestArea = Infinity;
            decoded.forEach((mask, i) => {
                if (!mask || !runsContain(mask.runs, point.x, point.y)) return;
                const area = runsArea(mask.runs);
                if (area < bestArea) {
                    bestArea = area;
                    bestId = owners[i].id;
                }
            });
            if (bestId !== null) props.onSelectTracklet(bestId);
        },
        [props],
    );

    const handleCanvasClick = useCallback(
        (event: ReactMouseEvent<HTMLCanvasElement>) => {
            const point = toFrame(event, drawing && props.method === "polygon");
            if (!point) return;
            if (!drawing) {
                if (props.tool === "review") void selectAt(point);
                return;
            }
            event.preventDefault();
            if (props.method === "point") {
                const negative = event.shiftKey || event.button === 2;
                props.onPromptPoint({
                    x: Math.round(point.x * 10) / 10,
                    y: Math.round(point.y * 10) / 10,
                    label: negative ? 0 : 1,
                });
            } else if (props.method === "polygon") {
                // An existing mask's outline is dragged, not clicked: adding a
                // point would land a stray vertex next to the hands.
                if (props.outline) return;
                if (event.button === 2) {
                    if (props.polygon.length >= 3) props.onPolygonClose();
                    return;
                }
                const layout = layoutRef.current;
                const first = props.polygon[0];
                if (first && layout && props.polygon.length >= 3) {
                    const distance =
                        Math.hypot(first.x - point.x, first.y - point.y) *
                        layout.scale;
                    if (distance <= CLOSE_POLYGON_RADIUS) {
                        props.onPolygonClose();
                        return;
                    }
                }
                props.onPolygonPoint({
                    x: Math.round(point.x * 10) / 10,
                    y: Math.round(point.y * 10) / 10,
                });
            }
        },
        [props, drawing, toFrame, selectAt],
    );

    const handleDoubleClick = useCallback(
        (event: ReactMouseEvent<HTMLCanvasElement>) => {
            if (!drawing || props.method !== "polygon") return;
            event.preventDefault();
            if (props.polygon.length >= 3) props.onPolygonClose();
        },
        [props, drawing],
    );

    const paintLiveSegment = useCallback(
        (from: FramePoint, to: FramePoint, mode: PaintMode) => {
            const canvas = canvasRef.current;
            const layout = layoutRef.current;
            const ctx = canvas?.getContext("2d");
            if (!canvas || !layout || !ctx) return;
            ctx.save();
            ctx.globalAlpha = 0.75;
            ctx.strokeStyle =
                mode === "add" ? PREVIEW_COLOR : CANDIDATE_ERASE_COLOR;
            ctx.lineCap = "round";
            ctx.lineJoin = "round";
            ctx.lineWidth = Math.max(1, props.brushSize * layout.scale);
            ctx.beginPath();
            ctx.moveTo(
                layout.x + from.x * layout.scale,
                layout.y + from.y * layout.scale,
            );
            const still = to.x === from.x && to.y === from.y;
            ctx.lineTo(
                layout.x + (to.x + (still ? 0.01 : 0)) * layout.scale,
                layout.y + to.y * layout.scale,
            );
            ctx.stroke();
            ctx.restore();
        },
        [props.brushSize],
    );

    const handlePointerDown = useCallback(
        (event: ReactPointerEvent<HTMLCanvasElement>) => {
            // Pan: middle-drag anywhere, right-drag where the right button has no
            // drawing meaning (so a create-mode right-click stays a prompt).
            if (
                event.button === 1 ||
                (event.button === 2 && !rightHasDrawMeaning)
            ) {
                event.preventDefault();
                panRef.current = {
                    pointerId: event.pointerId,
                    button: event.button,
                    lastX: event.clientX,
                    lastY: event.clientY,
                    moved: false,
                };
                setPanning(true);
                setCursor(null);
                event.currentTarget.setPointerCapture(event.pointerId);
                return;
            }
            // A box is a drag, not a click: start it here, commit it on release.
            if (drawing && props.method === "box") {
                if (event.button !== 0) return;
                const start = toFrame(event, true);
                if (!start) return;
                event.preventDefault();
                boxStartRef.current = start;
                setBoxRect({
                    x0: start.x,
                    y0: start.y,
                    x1: start.x,
                    y1: start.y,
                });
                return;
            }
            // Editing an existing mask's outline. The left button grabs (or
            // creates) a vertex to drag; the right button removes one, matching
            // the right-button vocabulary the brush already uses.
            const outline = props.outline;
            if (drawing && props.method === "polygon" && outline) {
                const layout = layoutRef.current;
                if (!layout) return;
                const point = toFrame(event, true);
                if (!point) return;
                const vertexRadius = VERTEX_HIT_RADIUS / layout.scale;
                if (event.button === 2) {
                    const doomed = nearestVertex(outline, point, vertexRadius);
                    if (!doomed) return;
                    const trimmed = removeVertex(
                        outline,
                        doomed.ring,
                        doomed.index,
                    );
                    if (!trimmed) return;
                    event.preventDefault();
                    props.onOutlineChange(trimmed);
                    return;
                }
                if (event.button !== 0) return;
                const grabbed = nearestVertex(outline, point, vertexRadius);
                let rings = outline;
                let ring = grabbed?.ring ?? -1;
                let index = grabbed?.index ?? -1;
                if (!grabbed) {
                    const edge = nearestEdge(
                        outline,
                        point,
                        EDGE_HIT_RADIUS / layout.scale,
                    );
                    // Only an existing edge can be split; a click on empty space
                    // is not how a missing blob is added (the brush is).
                    if (!edge) return;
                    rings = insertVertex(
                        outline,
                        edge.ring,
                        edge.index,
                        edge.at,
                    );
                    ring = edge.ring;
                    index = edge.index + 1;
                }
                event.preventDefault();
                dragRef.current = {
                    pointerId: event.pointerId,
                    ring,
                    index,
                    rings,
                };
                setDragRings(rings);
                event.currentTarget.setPointerCapture(event.pointerId);
                return;
            }
            if (!drawing || props.method !== "brush") return;
            if (event.button !== 0 && event.button !== 2) return;
            const point = toFrame(event);
            if (!point) return;
            event.preventDefault();
            const erase = event.shiftKey || event.button === 2;
            const mode: PaintMode = erase ? "erase" : props.paintMode;
            const raster = rasterRef.current!;
            raster.clear();
            raster.strokeSegment(point, point, props.brushSize);
            paintLiveSegment(point, point, mode);
            strokeRef.current = {
                mode,
                last: point,
                pointerId: event.pointerId,
            };
            event.currentTarget.setPointerCapture(event.pointerId);
        },
        [drawing, rightHasDrawMeaning, props, toFrame, paintLiveSegment],
    );

    const handlePointerMove = useCallback(
        (event: ReactPointerEvent<HTMLCanvasElement>) => {
            const layout = layoutRef.current;
            const canvas = canvasRef.current;
            if (!layout || !canvas) return;

            // Panning wins while a pan gesture is live: no cursor tracking, no
            // box drag, no stroke should be feeding off the same pointer.
            const pan = panRef.current;
            if (pan && pan.pointerId === event.pointerId) {
                const dx = event.clientX - pan.lastX;
                const dy = event.clientY - pan.lastY;
                pan.lastX = event.clientX;
                pan.lastY = event.clientY;
                pan.moved = true;
                if (dx !== 0 || dy !== 0) {
                    applyView(
                        panBy(
                            containerSize(),
                            frameSize(),
                            viewRef.current,
                            dx,
                            dy,
                        ),
                    );
                }
                return;
            }

            const drag = dragRef.current;
            if (drag && drag.pointerId === event.pointerId) {
                const rect = canvas.getBoundingClientRect();
                const raw = screenToFrame(
                    layout,
                    event.clientX - rect.left,
                    event.clientY - rect.top,
                );
                drag.rings = moveVertex(drag.rings, drag.ring, drag.index, {
                    x: Math.min(layout.width, Math.max(0, raw.x)),
                    y: Math.min(layout.height, Math.max(0, raw.y)),
                });
                setDragRings(drag.rings);
                return;
            }

            const rect = canvas.getBoundingClientRect();
            const raw: FramePoint = screenToFrame(
                layout,
                event.clientX - rect.left,
                event.clientY - rect.top,
            );
            if (drawing) setCursor(raw);
            if (boxStartRef.current) {
                const start = boxStartRef.current;
                const to: FramePoint = {
                    x: Math.min(layout.width, Math.max(0, raw.x)),
                    y: Math.min(layout.height, Math.max(0, raw.y)),
                };
                setBoxRect({
                    x0: Math.min(start.x, to.x),
                    y0: Math.min(start.y, to.y),
                    x1: Math.max(start.x, to.x),
                    y1: Math.max(start.y, to.y),
                });
                return;
            }
            const stroke = strokeRef.current;
            if (!stroke || stroke.pointerId !== event.pointerId) return;
            const point: FramePoint = {
                x: Math.min(layout.width, Math.max(0, raw.x)),
                y: Math.min(layout.height, Math.max(0, raw.y)),
            };
            rasterRef.current!.strokeSegment(
                stroke.last,
                point,
                props.brushSize,
            );
            paintLiveSegment(stroke.last, point, stroke.mode);
            stroke.last = point;
        },
        [
            drawing,
            applyView,
            containerSize,
            frameSize,
            props.brushSize,
            paintLiveSegment,
        ],
    );

    const finishStroke = useCallback(
        (event: ReactPointerEvent<HTMLCanvasElement>) => {
            const pan = panRef.current;
            if (pan && pan.pointerId === event.pointerId) {
                panRef.current = null;
                if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                    event.currentTarget.releasePointerCapture(event.pointerId);
                }
                setPanning(false);
                // A right-drag that actually panned is not also a right-click.
                if (pan.moved && pan.button === 2) {
                    swallowRightClickRef.current = true;
                }
                return;
            }
            const drag = dragRef.current;
            if (drag && drag.pointerId === event.pointerId) {
                dragRef.current = null;
                setDragRings(null);
                if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                    event.currentTarget.releasePointerCapture(event.pointerId);
                }
                props.onOutlineChange(drag.rings);
                return;
            }
            if (boxStartRef.current) {
                boxStartRef.current = null;
                const rect = boxRect;
                setBoxRect(null);
                // A stray press is not a box: require a usable area.
                if (rect && rect.x1 - rect.x0 >= 2 && rect.y1 - rect.y0 >= 2) {
                    props.onPromptBox?.(rect);
                }
                return;
            }
            const stroke = strokeRef.current;
            if (!stroke || stroke.pointerId !== event.pointerId) return;
            strokeRef.current = null;
            event.currentTarget.releasePointerCapture(event.pointerId);
            const rle = rasterRef.current!.toRle();
            if (rle) props.onStroke(rle, stroke.mode);
        },
        [props, boxRect],
    );

    const [viewport, setViewport] = useState({ w: 0, h: 0 });

    useEffect(() => {
        const element = wrapRef.current;
        if (!element) return;
        const observer = new ResizeObserver((entries) => {
            const rect = entries[0].contentRect;
            setViewport({ w: rect.width, h: rect.height });
        });
        observer.observe(element);
        setViewport({ w: element.clientWidth, h: element.clientHeight });
        return () => observer.disconnect();
    }, []);

    // Scroll to zoom, about whatever is under the cursor.
    //
    // This must be a native listener rather than an `onWheel` prop: React
    // attaches `wheel` at the root as a passive listener, so `preventDefault()`
    // from JSX is ignored and the page scrolls while you zoom.
    useEffect(() => {
        const element = wrapRef.current;
        if (!element) return;
        const onWheel = (event: WheelEvent) => {
            const layout = layoutRef.current;
            if (!layout) return;
            event.preventDefault();
            const bounds = element.getBoundingClientRect();
            const container = { w: bounds.width, h: bounds.height };
            const current = viewRef.current;
            applyView(
                zoomAbout(
                    container,
                    { w: layout.width, h: layout.height },
                    current,
                    clampZoom(
                        current.zoom *
                            wheelZoomFactor(event.deltaY, event.deltaMode),
                    ),
                    {
                        x: event.clientX - bounds.left,
                        y: event.clientY - bounds.top,
                    },
                ),
            );
        };
        element.addEventListener("wheel", onWheel, { passive: false });
        return () => element.removeEventListener("wheel", onWheel);
    }, [applyView]);

    useEffect(() => {
        const cache = cacheRef.current;
        if (!cache) return;
        const total = props.clip.frameCount;
        const wanted: number[] = [];
        for (let d = 1; d <= 15; d++) {
            if (props.frameIndex + d < total) wanted.push(props.frameIndex + d);
            if (props.frameIndex - d >= 0) wanted.push(props.frameIndex - d);
        }
        cache.preload(wanted);
    }, [props.frameIndex, props.clip]);

    useEffect(() => {
        const cache = maskCacheRef.current;
        if (!cache) return;
        const total = props.clip.frameCount;
        const visible = maskTracklets();

        const requests: MaskRequest[] = [];
        const MAX_PREFETCH_MASKS = 64;
        for (let d = 1; d <= 8 && requests.length < MAX_PREFETCH_MASKS; d++) {
            const index = props.frameIndex + d;
            if (index >= total) break;
            for (const tracklet of visible) {
                if (requests.length >= MAX_PREFETCH_MASKS) break;
                const payload = props.clip.rawMaskAt(tracklet, index);
                if (!payload) continue;
                requests.push({
                    trackletId: tracklet.id,
                    frameIndex: index,
                    payload,
                });
            }
        }
        if (requests.length === 0) return;
        void cache.resolveBatch(requests).catch(() => {});
    }, [props.frameIndex, props.clip, maskTracklets]);

    useEffect(() => {
        const cache = maskCacheRef.current;
        if (!cache) return;
        const tracklet = props.clip.tracklets.find(
            (t) => t.id === props.selectedTrackletId,
        );
        if (!tracklet) return;

        const requests: MaskRequest[] = [];
        for (let index = 0; index < props.clip.frameCount; index++) {
            const payload = props.clip.rawMaskAt(tracklet, index);
            if (!payload) continue;
            requests.push({
                trackletId: tracklet.id,
                frameIndex: index,
                payload,
            });
        }

        let cancelled = false;
        const CHUNK_SIZE = 64;
        const pump = async () => {
            for (
                let offset = 0;
                offset < requests.length && !cancelled;
                offset += CHUNK_SIZE
            ) {
                await cache.resolveBatch(
                    requests.slice(offset, offset + CHUNK_SIZE),
                );
            }
        };
        void pump().catch(() => {});

        return () => {
            cancelled = true;
        };
    }, [props.clip, props.selectedTrackletId]);

    // Render the current frame and its mask overlay.
    useEffect(() => {
        let cancelled = false;

        const draw = async (attempt = 0) => {
            const canvas = canvasRef.current;
            const maskRenderer = maskRef.current;
            const edgeRenderer = edgeRef.current;
            if (!canvas || !maskRenderer || !edgeRenderer) return;

            // Measure the wrapper here rather than trusting the observer's state.
            // Setting canvas.width clears the canvas, so a stale 0x0 reading would
            // wipe the frame and then bail out, leaving a blank panel and no error.
            const bounds = wrapRef.current?.getBoundingClientRect();
            const width = Math.round(bounds?.width ?? 0);
            const height = Math.round(bounds?.height ?? 0);
            if (width === 0 || height === 0) {
                // Layout has not settled. Retry for a second, then give up rather
                // than spinning on animation frames forever.
                if (attempt < 60 && !cancelled) {
                    retryRef.current = requestAnimationFrame(() => {
                        void draw(attempt + 1);
                    });
                }
                return;
            }

            const dpr = window.devicePixelRatio || 1;
            const backingWidth = Math.max(1, Math.round(width * dpr));
            const backingHeight = Math.max(1, Math.round(height * dpr));
            if (canvas.width !== backingWidth) canvas.width = backingWidth;
            if (canvas.height !== backingHeight) canvas.height = backingHeight;

            const ctx = canvas.getContext("2d");
            if (!ctx) return;

            let frame: ImageBitmap | null = null;
            try {
                frame = await cacheRef.current!.get(props.frameIndex);
            } catch {
                frame = null;
            }
            if (cancelled) return;

            const frameWidth = frame ? frame.width : props.clip.width;
            const frameHeight = frame ? frame.height : props.clip.height;
            const nextLayout = frameLayout(
                { w: width, h: height },
                { w: frameWidth, h: frameHeight },
                view,
            );
            const scale = nextLayout.scale;
            const drawWidth = frameWidth * scale;
            const drawHeight = frameHeight * scale;
            const drawX = nextLayout.x;
            const drawY = nextLayout.y;
            layoutRef.current = nextLayout;
            setLayout((current) =>
                current &&
                current.x === nextLayout.x &&
                current.y === nextLayout.y &&
                current.scale === nextLayout.scale &&
                current.width === nextLayout.width &&
                current.height === nextLayout.height
                    ? current
                    : nextLayout,
            );

            // Only the visible slice of the source is ever resampled. Zoomed in,
            // a full-frame drawImage would touch every source pixel on every
            // repaint, and panning repaints on every pointer move.
            const slice = visibleSlice(nextLayout, { w: width, h: height });
            // Resample only when shrinking. Magnified, both the frame and the mask
            // show their true pixels: a blurred boundary is the wrong thing to put
            // in front of someone deciding whether a mask is accurate.
            const smooth = slice ? slice.smooth : false;
            const blit = (
                source: CanvasImageSource,
                alpha: number,
                smooth: boolean,
            ) => {
                if (!slice) return;
                ctx.save();
                ctx.globalAlpha = alpha;
                ctx.imageSmoothingEnabled = smooth;
                ctx.drawImage(
                    source,
                    slice.sx,
                    slice.sy,
                    slice.sw,
                    slice.sh,
                    slice.dx,
                    slice.dy,
                    slice.dw,
                    slice.dh,
                );
                ctx.restore();
            };

            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            ctx.fillStyle = "#000";
            ctx.fillRect(0, 0, width, height);

            if (frame) {
                blit(frame, 1, smooth);
            } else {
                ctx.fillStyle = "#1b1f24";
                ctx.fillRect(drawX, drawY, drawWidth, drawHeight);
                ctx.fillStyle = "#9aa4af";
                ctx.font = "14px system-ui";
                ctx.fillText("Frame unavailable", drawX + 12, drawY + 24);
            }
            paintedFrameRef.current = props.frameIndex;

            const cache = maskCacheRef.current!;
            const visible = maskTracklets();

            const requests: MaskRequest[] = [];
            const requestTracklets: Tracklet[] = [];
            for (const tracklet of visible) {
                if (tracklet.id === props.editingTrackletId) continue;
                const payload = props.clip.rawMaskAt(
                    tracklet,
                    props.frameIndex,
                );
                if (!payload) continue;
                requests.push({
                    trackletId: tracklet.id,
                    frameIndex: props.frameIndex,
                    payload,
                });
                requestTracklets.push(tracklet);
            }

            const decodedList =
                requests.length > 0 ? await cache.resolveBatch(requests) : [];

            if (cancelled || paintedFrameRef.current !== props.frameIndex)
                return;

            // Fill every object at the overlay opacity — the selected one last,
            // in the reserved colour, so "the one you are working on" stands out
            // without hiding what surrounds it. The outline is drawn into its own
            // buffer and blitted opaque afterwards, so it reads as a crisp edge
            // no matter how faint the fill is.
            const boundaryWidth = boundaryWidthFor(
                props.clip.width,
                props.clip.height,
            );
            let selectedMask: DecodedMask | null = null;
            maskRenderer.clear();
            edgeRenderer.clear();
            for (let i = 0; i < requestTracklets.length; i++) {
                const decoded = decodedList[i];
                if (!decoded) continue;
                const tracklet = requestTracklets[i];
                if (tracklet.id === props.selectedTrackletId) {
                    selectedMask = decoded;
                    continue;
                }
                maskRenderer.drawRuns(decoded.runs, tracklet.color);
                edgeRenderer.strokeRuns(
                    decoded.runs,
                    tracklet.color,
                    boundaryWidth,
                );
            }
            if (selectedMask) {
                maskRenderer.drawRuns(selectedMask.runs, SELECTED_COLOR);
                // Select mode hands this outline to the marching-ants layer, so
                // the two never fight over the same edge.
                if (props.tool !== "review") {
                    edgeRenderer.strokeRuns(
                        selectedMask.runs,
                        SELECTED_COLOR,
                        boundaryWidth,
                    );
                }
            }

            blit(maskRenderer.canvasElement, props.maskOpacity, smooth);
            blit(edgeRenderer.canvasElement, 1, smooth);

            if (props.tool === "review") return;

            const overlays: { mask: DecodedMask | null; color: string }[] = [
                { mask: props.draft, color: PREVIEW_COLOR },
                {
                    mask: props.candidate,
                    color:
                        props.candidateColor ??
                        (props.paintMode === "add"
                            ? CANDIDATE_ADD_COLOR
                            : CANDIDATE_ERASE_COLOR),
                },
            ];
            for (const { mask, color } of overlays) {
                if (!mask) continue;
                maskRenderer.clear();
                maskRenderer.drawRuns(mask.runs, color);
                blit(
                    maskRenderer.canvasElement,
                    Math.max(0.6, props.maskOpacity),
                    smooth,
                );
            }

            if (!drawing || props.method === "brush") return;

            for (const point of props.prompt) {
                const px = drawX + point.x * scale;
                const py = drawY + point.y * scale;
                ctx.beginPath();
                ctx.arc(px, py, 6, 0, Math.PI * 2);
                ctx.fillStyle =
                    point.label === 1 ? POSITIVE_COLOR : NEGATIVE_COLOR;
                ctx.fill();
                ctx.lineWidth = 2;
                ctx.strokeStyle = "#fff";
                ctx.stroke();
                if (point.label === 0) {
                    ctx.beginPath();
                    ctx.moveTo(px - 3, py);
                    ctx.lineTo(px + 3, py);
                    ctx.strokeStyle = "#fff";
                    ctx.lineWidth = 1.5;
                    ctx.stroke();
                }
            }
        };

        void draw();
        return () => {
            cancelled = true;
            if (retryRef.current !== null) {
                cancelAnimationFrame(retryRef.current);
                retryRef.current = null;
            }
        };
    }, [
        viewport,
        view,
        props.clip,
        props.frameIndex,
        props.selectedTrackletId,
        props.maskOpacity,
        props.tool,
        props.method,
        props.paintMode,
        props.prompt,
        props.draft,
        props.candidate,
        props.candidateColor,
        props.editingTrackletId,
        maskTracklets,
    ]);

    // Select mode: the selected object's outline becomes marching ants.
    //
    // The outline is stroked as a real path with a dash pattern, so the dashes
    // are segments *of the boundary* and `lineDashOffset` walks them along it:
    // they follow a curve, turn a corner and circulate a hole, which a pattern
    // sliding in one direction cannot do. The path is traced once per (clip,
    // object, frame) and cached, so each animation frame is a single vector
    // stroke on a small transparent layer — no frame redraw, no pixel work. The
    // loop runs only while Select has an object with a mask on this frame; every
    // other tool clears the layer and stops it.
    useEffect(() => {
        const canvas = antsRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        const dpr = window.devicePixelRatio || 1;
        const backingWidth = Math.max(1, Math.round(viewport.w * dpr));
        const backingHeight = Math.max(1, Math.round(viewport.h * dpr));
        if (canvas.width !== backingWidth || canvas.height !== backingHeight) {
            canvas.width = backingWidth;
            canvas.height = backingHeight;
        }

        let raf = 0;
        const stop = () => {
            if (raf) cancelAnimationFrame(raf);
            raf = 0;
            ctx.setTransform(1, 0, 0, 1, 0, 0);
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        };

        const tracklet =
            props.tool === "review"
                ? props.clip.tracklets.find(
                      (t) => t.id === props.selectedTrackletId,
                  )
                : undefined;
        const payload = tracklet
            ? props.clip.rawMaskAt(tracklet, props.frameIndex)
            : null;
        const layout =
            payload && viewport.w > 0 && viewport.h > 0
                ? frameLayout(
                      viewport,
                      { w: props.clip.width, h: props.clip.height },
                      view,
                  )
                : null;
        if (
            !tracklet ||
            !payload ||
            !layout ||
            !visibleSlice(layout, viewport)
        ) {
            stop();
            return;
        }

        // Band width and dash pitch are frame pixels, so the ants thicken and
        // lengthen with the outline as it is zoomed, exactly as the solid
        // boundaries do. A dash of four band widths reads clearly without
        // turning the outline into a dotted line.
        const thickness = boundaryWidthFor(props.clip.width, props.clip.height);
        const dash = thickness * 4;
        const cycle = dash * 2;

        // The path outlives pans and zooms; only a changed clip, object or frame
        // makes it stale.
        const cached = antsKeyRef.current;
        const fresh =
            cached !== null &&
            cached.clip === props.clip &&
            cached.trackletId === tracklet.id &&
            cached.frameIndex === props.frameIndex;
        let path: Path2D | null = fresh ? antsPathRef.current : null;

        let cancelled = false;
        let last = 0;
        const step = (now: number) => {
            if (cancelled || !path) return;
            // One dash cycle per `ANTS_PERIOD`, so the ants travel at the same
            // rate at any zoom and on any refresh rate.
            const elapsed = last === 0 ? 0 : Math.min(64, now - last);
            last = now;
            antsPhaseRef.current =
                (antsPhaseRef.current + (elapsed / ANTS_PERIOD) * cycle) %
                cycle;

            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            ctx.clearRect(0, 0, viewport.w, viewport.h);
            ctx.save();
            // Frame coordinates -> screen, so the stroke is crisp at any zoom.
            ctx.translate(layout.x, layout.y);
            ctx.scale(layout.scale, layout.scale);
            ctx.lineWidth = thickness;
            ctx.strokeStyle = SELECTED_COLOR;
            ctx.lineJoin = "round";
            ctx.setLineDash([dash, dash]);
            // Negative, so the dashes advance along the path rather than against
            // it — the direction of travel follows the boundary.
            ctx.lineDashOffset = -antsPhaseRef.current;
            ctx.stroke(path);
            ctx.restore();
            raf = requestAnimationFrame(step);
        };

        if (path) {
            raf = requestAnimationFrame(step);
        } else {
            const maskCache = maskCacheRef.current;
            if (!maskCache) {
                stop();
                return;
            }
            void maskCache
                .resolveBatch([
                    {
                        trackletId: tracklet.id,
                        frameIndex: props.frameIndex,
                        payload,
                    },
                ])
                .then((decoded) => {
                    if (cancelled) return;
                    const mask = decoded[0];
                    if (!mask) {
                        stop();
                        return;
                    }
                    const built = ringsToPath(
                        outlineFromRuns(mask.runs, mask.width, mask.height),
                    );
                    antsPathRef.current = built;
                    antsKeyRef.current = {
                        clip: props.clip,
                        trackletId: tracklet.id,
                        frameIndex: props.frameIndex,
                    };
                    path = built;
                    raf = requestAnimationFrame(step);
                })
                .catch(() => stop());
        }

        return () => {
            cancelled = true;
            stop();
        };
    }, [
        props.tool,
        props.selectedTrackletId,
        props.frameIndex,
        props.clip,
        view,
        viewport,
    ]);

    let hint: string | null = null;
    if (props.tool === "propagate") {
        hint = props.promptHint ?? null;
    } else if (drawing) {
        if (props.method === "point") {
            hint =
                props.prompt.length === 0
                    ? (props.promptHint ?? "Click the object")
                    : `${props.prompt.length} point${props.prompt.length === 1 ? "" : "s"}`;
            if (props.box) hint += " · box prompt active";
        } else if (props.method === "box") {
            hint = boxRect
                ? `Release to segment inside the box (${Math.round(boxRect.x1 - boxRect.x0)}×${Math.round(boxRect.y1 - boxRect.y0)} px)`
                : "Drag a box around the object";
        } else if (props.method === "text") {
            hint = "Type a class name and press Enter";
        } else if (props.method === "polygon") {
            hint = props.outline
                ? "Drag a vertex · click an edge to add one · right-click to remove"
                : props.polygon.length === 0
                  ? "Click to place points · Enter closes"
                  : `${props.polygon.length} point${props.polygon.length === 1 ? "" : "s"} · Enter closes`;
        } else {
            hint = `Drag to paint${props.paintMode === "erase" ? " (erasing)" : ""}`;
        }
    }

    const brushCursor =
        drawing && props.method === "brush" && cursor && layout
            ? {
                  x: layout.x + cursor.x * layout.scale,
                  y: layout.y + cursor.y * layout.scale,
                  size: Math.max(2, props.brushSize * layout.scale),
              }
            : null;
    const toScreen = (p: FramePoint) =>
        layout
            ? `${layout.x + p.x * layout.scale},${layout.y + p.y * layout.scale}`
            : "";
    /** What the overlay draws: the live drag if one is running, else the props. */
    const outlineRings = dragRings ?? props.outline;

    return (
        <div className={styles.panel}>
            <div
                ref={wrapRef}
                className={`${styles.canvasWrap} ${drawing ? styles.prompting : ""} ${
                    drawing && props.method === "brush" ? styles.brushing : ""
                } ${drawing || view.zoom <= MIN_ZOOM ? "" : styles.pannable} ${
                    panning ? styles.panning : ""
                }`}
            >
                <canvas
                    ref={canvasRef}
                    className={styles.canvas}
                    onClick={handleCanvasClick}
                    onDoubleClick={handleDoubleClick}
                    onPointerDown={handlePointerDown}
                    onPointerMove={handlePointerMove}
                    onPointerUp={finishStroke}
                    onPointerCancel={finishStroke}
                    onPointerLeave={() => setCursor(null)}
                    title="Scroll to zoom · middle-drag to pan"
                    onContextMenu={(event) => {
                        // The canvas owns the right button: an OS menu would
                        // otherwise pop up in the middle of a right-drag pan.
                        event.preventDefault();
                        if (swallowRightClickRef.current) {
                            // That right-drag was a pan, not a click.
                            swallowRightClickRef.current = false;
                            return;
                        }
                        if (!drawing) return;
                        if (props.method !== "brush") handleCanvasClick(event);
                    }}
                />
                <canvas
                    ref={antsRef}
                    className={styles.ants}
                    aria-hidden="true"
                />
                {drawing && layout && (
                    <svg
                        className={styles.overlay}
                        aria-hidden="true"
                        width={viewport.w}
                        height={viewport.h}
                    >
                        {props.box && (
                            <rect
                                x={layout.x + props.box.x0 * layout.scale}
                                y={layout.y + props.box.y0 * layout.scale}
                                width={
                                    (props.box.x1 - props.box.x0) * layout.scale
                                }
                                height={
                                    (props.box.y1 - props.box.y0) * layout.scale
                                }
                                fill="none"
                                stroke={POSITIVE_COLOR}
                                strokeWidth={1.5}
                                strokeDasharray="3 3"
                            />
                        )}
                        {props.method === "box" && boxRect && (
                            <rect
                                x={layout.x + boxRect.x0 * layout.scale}
                                y={layout.y + boxRect.y0 * layout.scale}
                                width={(boxRect.x1 - boxRect.x0) * layout.scale}
                                height={
                                    (boxRect.y1 - boxRect.y0) * layout.scale
                                }
                                fill="none"
                                stroke={PREVIEW_COLOR}
                                strokeWidth={2}
                                strokeDasharray="6 4"
                            />
                        )}
                        {props.method === "polygon" &&
                            props.polygon.length > 0 && (
                                <>
                                    <polyline
                                        className={styles.polygonLine}
                                        points={props.polygon
                                            .map(toScreen)
                                            .join(" ")}
                                    />
                                    {cursor && (
                                        <line
                                            className={styles.rubberBand}
                                            x1={
                                                toScreen(
                                                    props.polygon[
                                                        props.polygon.length - 1
                                                    ],
                                                ).split(",")[0]
                                            }
                                            y1={
                                                toScreen(
                                                    props.polygon[
                                                        props.polygon.length - 1
                                                    ],
                                                ).split(",")[1]
                                            }
                                            x2={
                                                layout.x +
                                                cursor.x * layout.scale
                                            }
                                            y2={
                                                layout.y +
                                                cursor.y * layout.scale
                                            }
                                        />
                                    )}
                                    {props.polygon.map((p, i) => (
                                        <circle
                                            key={i}
                                            className={
                                                i === 0 &&
                                                props.polygon.length >= 3
                                                    ? styles.vertexFirst
                                                    : styles.vertex
                                            }
                                            cx={layout.x + p.x * layout.scale}
                                            cy={layout.y + p.y * layout.scale}
                                            r={
                                                i === 0 &&
                                                props.polygon.length >= 3
                                                    ? 6
                                                    : 4
                                            }
                                        />
                                    ))}
                                </>
                            )}
                        {outlineRings?.map((ring, ringIndex) => (
                            <g key={ringIndex}>
                                <polygon
                                    className={styles.polygonLine}
                                    points={ring.map(toScreen).join(" ")}
                                />
                                {ring.map((p, i) => (
                                    <circle
                                        key={i}
                                        className={styles.vertex}
                                        cx={layout.x + p.x * layout.scale}
                                        cy={layout.y + p.y * layout.scale}
                                        r={4}
                                    />
                                ))}
                            </g>
                        ))}
                        {brushCursor && (
                            <circle
                                className={
                                    props.paintMode === "add"
                                        ? styles.brushCursor
                                        : styles.brushCursorErase
                                }
                                cx={brushCursor.x}
                                cy={brushCursor.y}
                                r={brushCursor.size / 2}
                            />
                        )}
                    </svg>
                )}
                {hint && (
                    <div className={styles.promptHint} aria-live="polite">
                        {hint}
                    </div>
                )}
            </div>

            <div className={styles.controls}>
                <button
                    type="button"
                    className={`btn btnIcon ${styles.play}`}
                    onClick={props.onPlayToggle}
                    title={props.playing ? "Pause (Space)" : "Play (Space)"}
                    aria-label={props.playing ? "Pause" : "Play"}
                >
                    <Icon name={props.playing ? "pause" : "play"} size={16} />
                </button>
                <button
                    type="button"
                    className="btn btnIcon"
                    onClick={() => props.onStep(-1)}
                    title="Step back one frame (←)"
                    aria-label="Step back one frame"
                >
                    <Icon name="stepBack" size={16} />
                </button>
                <button
                    type="button"
                    className="btn btnIcon"
                    onClick={() => props.onStep(1)}
                    title="Step forward one frame (→)"
                    aria-label="Step forward one frame"
                >
                    <Icon name="stepForward" size={16} />
                </button>

                <input
                    type="range"
                    className={styles.slider}
                    min={0}
                    max={Math.max(0, props.clip.frameCount - 1)}
                    step={1}
                    value={props.frameIndex}
                    onChange={(event) =>
                        props.onFrameChange(Number(event.target.value))
                    }
                />

                <span className={styles.timecode}>
                    {props.frameIndex + 1} / {props.clip.frameCount} ·{" "}
                    {formatTimecode(props.frameIndex, props.clip.fps)}
                </span>

                <label className={styles.opacityGroup}>
                    Overlay
                    <input
                        type="range"
                        min={0}
                        max={1}
                        step={0.05}
                        value={props.maskOpacity}
                        onChange={(event) =>
                            props.onMaskOpacityChange(
                                Number(event.target.value),
                            )
                        }
                    />
                    {Math.round(props.maskOpacity * 100)}%
                </label>

                <div className={styles.zoomGroup}>
                    <button
                        type="button"
                        className="btn btnIcon btnSmall"
                        onClick={() => zoomBy(1 / 1.5)}
                        disabled={view.zoom <= MIN_ZOOM}
                        title="Zoom out"
                        aria-label="Zoom out"
                    >
                        <Icon name="zoomOut" size={14} />
                    </button>
                    <span className={styles.zoomValue}>
                        {Math.round(view.zoom * 100)}%
                    </span>
                    <button
                        type="button"
                        className="btn btnIcon btnSmall"
                        onClick={() => zoomBy(1.5)}
                        disabled={view.zoom >= MAX_ZOOM}
                        title="Zoom in"
                        aria-label="Zoom in"
                    >
                        <Icon name="zoomIn" size={14} />
                    </button>
                    <button
                        type="button"
                        className="btn btnSmall"
                        onClick={resetView}
                        disabled={view.zoom === FIT_VIEW.zoom}
                        title="Fit the whole frame"
                    >
                        <Icon name="fit" size={14} />
                        Fit
                    </button>
                </div>
            </div>
        </div>
    );
}
