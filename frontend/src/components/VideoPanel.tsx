import { useCallback, useEffect, useRef, useState } from "react";
import type {
    MouseEvent as ReactMouseEvent,
    PointerEvent as ReactPointerEvent,
} from "react";
import type { Clip } from "../lib/clip";
import type { FrameSource } from "../lib/frames";
import { FrameCache } from "../lib/frameCache";
import { MaskRenderer } from "../lib/mask";
import { MaskCache, type MaskRequest } from "../lib/maskApi";
import { formatTimecode } from "../lib/format";
import {
    RasterCanvas,
    runsArea,
    runsContain,
    type FramePoint,
    type PaintMode,
} from "../lib/raster";
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

interface VideoPanelProps {
    clip: Clip;
    frames: FrameSource;
    frameIndex: number;
    playing: boolean;
    selectedTrackletId: number | null;
    showAllMasks: boolean;
    maskOpacity: number;
    onFrameChange: (frame: number) => void;
    onPlayToggle: () => void;
    onStep: (delta: number) => void;
    onShowAllMasksChange: (value: boolean) => void;
    onMaskOpacityChange: (value: number) => void;
    tool: Tool;
    method: DrawMethod;
    paintMode: PaintMode;
    brushSize: number;
    prompt: PromptPoint[];
    polygon: FramePoint[];
    draft: DecodedMask | null;
    candidate: DecodedMask | null;
    editingTrackletId: number | null;
    onPromptPoint: (point: PromptPoint) => void;
    /** A box drag finished; the reviewer segments the object inside it. */
    onPromptBox?: (rect: PromptBox) => void;
    /** A box that is part of the current prompt (shown so it can be cleared). */
    box?: PromptBox | null;
    onPolygonPoint: (point: FramePoint) => void;
    onPolygonClose: () => void;
    onStroke: (stroke: RawRle, mode: PaintMode) => void;
    onSelectTracklet: (id: number) => void;
    promptHint?: string;
}

export function VideoPanel(props: VideoPanelProps) {
    const wrapRef = useRef<HTMLDivElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
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

    const [cursor, setCursor] = useState<FramePoint | null>(null);
    //: The box being dragged (box prompt): live while dragging, committed on release.
    const [boxRect, setBoxRect] = useState<PromptBox | null>(null);
    const boxStartRef = useRef<FramePoint | null>(null);

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
        [
            drawing,
            rightHasDrawMeaning,
            props.method,
            props.paintMode,
            props.brushSize,
            toFrame,
            paintLiveSegment,
        ],
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
        const visible = props.showAllMasks
            ? props.clip.tracklets
            : props.clip.tracklets.filter(
                  (t) => t.id === props.selectedTrackletId,
              );

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
    }, [
        props.frameIndex,
        props.clip,
        props.selectedTrackletId,
        props.showAllMasks,
    ]);

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
            if (!canvas || !maskRenderer) return;

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
            const sourceX = Math.max(0, -drawX / scale);
            const sourceY = Math.max(0, -drawY / scale);
            const sourceWidth = Math.min(frameWidth - sourceX, width / scale);
            const sourceHeight = Math.min(
                frameHeight - sourceY,
                height / scale,
            );
            const canBlit = scale > 0 && sourceWidth > 0 && sourceHeight > 0;
            // Resample only when shrinking. Magnified, both the frame and the mask
            // show their true pixels: a blurred boundary is the wrong thing to put
            // in front of someone deciding whether a mask is accurate.
            const smooth = scale < 1;
            const blit = (
                source: CanvasImageSource,
                alpha: number,
                smooth: boolean,
            ) => {
                if (!canBlit) return;
                ctx.save();
                ctx.globalAlpha = alpha;
                ctx.imageSmoothingEnabled = smooth;
                ctx.drawImage(
                    source,
                    sourceX,
                    sourceY,
                    sourceWidth,
                    sourceHeight,
                    drawX + sourceX * scale,
                    drawY + sourceY * scale,
                    sourceWidth * scale,
                    sourceHeight * scale,
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
            const visible = props.showAllMasks
                ? props.clip.tracklets
                : props.clip.tracklets.filter(
                      (t) => t.id === props.selectedTrackletId,
                  );

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

            maskRenderer.clear();
            for (let i = 0; i < requestTracklets.length; i++) {
                const decoded = decodedList[i];
                if (!decoded) continue;
                maskRenderer.drawRuns(decoded.runs, requestTracklets[i].color);
            }

            blit(maskRenderer.canvasElement, props.maskOpacity, smooth);

            if (props.tool === "review") return;

            const overlays: { mask: DecodedMask | null; color: string }[] = [
                { mask: props.draft, color: PREVIEW_COLOR },
                {
                    mask: props.candidate,
                    color:
                        props.paintMode === "add"
                            ? CANDIDATE_ADD_COLOR
                            : CANDIDATE_ERASE_COLOR,
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
        props.showAllMasks,
        props.maskOpacity,
        props.tool,
        props.method,
        props.paintMode,
        props.prompt,
        props.draft,
        props.candidate,
        props.editingTrackletId,
    ]);

    let hint: string | null = null;
    if (props.tool === "propagate") {
        hint = props.promptHint ?? null;
    } else if (drawing) {
        if (props.method === "point") {
            hint =
                props.prompt.length === 0
                    ? (props.promptHint ??
                      "Click the object to segment it · Shift-click or right-click to exclude a region")
                    : `${props.prompt.length} point${props.prompt.length === 1 ? "" : "s"} · keep clicking to refine`;
            if (props.box)
                hint +=
                    " · the box prompt is still active (clear it to use clicks alone)";
        } else if (props.method === "box") {
            hint = boxRect
                ? `Release to segment inside the box (${Math.round(boxRect.x1 - boxRect.x0)}×${Math.round(boxRect.y1 - boxRect.y0)} px)`
                : "Drag a box around the object";
        } else if (props.method === "text") {
            hint = "Type a class name in the prompt bar and press Enter";
        } else if (props.method === "polygon") {
            hint =
                props.polygon.length === 0
                    ? "Click to place polygon vertices · double-click, right-click or Enter closes it"
                    : `${props.polygon.length} vert${props.polygon.length === 1 ? "ex" : "ices"} · click the first vertex, double-click or press Enter to close`;
        } else {
            hint = `Drag to paint${props.paintMode === "erase" ? " (erasing)" : ""} · Shift-drag or right-drag erases · [ and ] change the brush size`;
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
                    className="btn"
                    onClick={props.onPlayToggle}
                    title="Space"
                >
                    {props.playing ? "Pause" : "Play"}
                </button>
                <button
                    type="button"
                    className="btn"
                    onClick={() => props.onStep(-1)}
                    title="←"
                >
                    ◀
                </button>
                <button
                    type="button"
                    className="btn"
                    onClick={() => props.onStep(1)}
                    title="→"
                >
                    ▶
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

                <label className={styles.check}>
                    <input
                        type="checkbox"
                        checked={props.showAllMasks}
                        onChange={(event) =>
                            props.onShowAllMasksChange(event.target.checked)
                        }
                    />
                    Show all masks
                </label>

                <div className={styles.zoomGroup}>
                    <button
                        type="button"
                        className="btn"
                        onClick={() => zoomBy(1 / 1.5)}
                        disabled={view.zoom <= MIN_ZOOM}
                        title="Zoom out"
                    >
                        −
                    </button>
                    <span className={styles.zoomValue}>
                        {Math.round(view.zoom * 100)}%
                    </span>
                    <button
                        type="button"
                        className="btn"
                        onClick={() => zoomBy(1.5)}
                        disabled={view.zoom >= MAX_ZOOM}
                        title="Zoom in"
                    >
                        +
                    </button>
                    <button
                        type="button"
                        className="btn"
                        onClick={resetView}
                        disabled={view.zoom === FIT_VIEW.zoom}
                        title="Fit the whole frame"
                    >
                        Fit
                    </button>
                </div>
            </div>
        </div>
    );
}
