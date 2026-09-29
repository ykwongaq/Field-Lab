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
import type { DrawMethod, Tool } from "./Toolbar";
import styles from "./VideoPanel.module.css";

interface FrameLayout {
    x: number;
    y: number;
    scale: number;
    width: number;
    height: number;
}

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
    onPolygonPoint: (point: FramePoint) => void;
    onPolygonClose: () => void;
    onStroke: (stroke: RawRle, mode: PaintMode) => void;
    onSelectTracklet: (id: number) => void;
    promptHint?: string;
}

export function VideoPanel(props: VideoPanelProps) {
    const wrapRef = useRef<HTMLDivElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);

    const cacheRef = useRef<FrameCache | null>(null);
    if (!cacheRef.current) {
        cacheRef.current = new FrameCache(props.frames);
    }
    const maskRef = useRef<MaskRenderer | null>(null);
    if (!maskRef.current) {
        maskRef.current = new MaskRenderer(props.clip.width, props.clip.height);
    }
    const maskCacheRef = useRef<MaskCache | null>(null);
    if (!maskCacheRef.current) {
        maskCacheRef.current = new MaskCache();
    }

    const paintedFrameRef = useRef(-1);

    const layoutRef = useRef<FrameLayout | null>(null);
    const [layout, setLayout] = useState<FrameLayout | null>(null);

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

    const drawing = props.tool === "addMask" || props.tool === "editMask";

    const toFrame = useCallback(
        (
            event: { clientX: number; clientY: number },
            clamp = false,
        ): FramePoint | null => {
            const layout = layoutRef.current;
            const canvas = canvasRef.current;
            if (!layout || !canvas) return null;
            const rect = canvas.getBoundingClientRect();
            const x = (event.clientX - rect.left - layout.x) / layout.scale;
            const y = (event.clientY - rect.top - layout.y) / layout.scale;
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
            if (props.method === "sam") {
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
            const rect = canvas.getBoundingClientRect();
            const raw: FramePoint = {
                x: (event.clientX - rect.left - layout.x) / layout.scale,
                y: (event.clientY - rect.top - layout.y) / layout.scale,
            };
            if (drawing) setCursor(raw);
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
        [drawing, props.brushSize, paintLiveSegment],
    );

    const finishStroke = useCallback(
        (event: ReactPointerEvent<HTMLCanvasElement>) => {
            const stroke = strokeRef.current;
            if (!stroke || stroke.pointerId !== event.pointerId) return;
            strokeRef.current = null;
            event.currentTarget.releasePointerCapture(event.pointerId);
            const rle = rasterRef.current!.toRle();
            if (rle) props.onStroke(rle, stroke.mode);
        },
        [props],
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

        const draw = async () => {
            const canvas = canvasRef.current;
            const maskRenderer = maskRef.current;
            if (!canvas || !maskRenderer) return;

            const dpr = window.devicePixelRatio || 1;
            const backingWidth = Math.max(1, Math.round(viewport.w * dpr));
            const backingHeight = Math.max(1, Math.round(viewport.h * dpr));
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
            if (viewport.w === 0 || viewport.h === 0) return;

            const frameWidth = frame ? frame.width : props.clip.width;
            const frameHeight = frame ? frame.height : props.clip.height;
            const scale = Math.min(
                viewport.w / frameWidth,
                viewport.h / frameHeight,
            );
            const drawWidth = frameWidth * scale;
            const drawHeight = frameHeight * scale;
            const drawX = (viewport.w - drawWidth) / 2;
            const drawY = (viewport.h - drawHeight) / 2;
            const nextLayout: FrameLayout = {
                x: drawX,
                y: drawY,
                scale,
                width: frameWidth,
                height: frameHeight,
            };
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

            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            ctx.fillStyle = "#000";
            ctx.fillRect(0, 0, viewport.w, viewport.h);

            if (frame) {
                ctx.drawImage(frame, drawX, drawY, drawWidth, drawHeight);
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

            ctx.save();
            ctx.globalAlpha = props.maskOpacity;
            ctx.drawImage(
                maskRenderer.canvasElement,
                drawX,
                drawY,
                drawWidth,
                drawHeight,
            );
            ctx.restore();

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
                ctx.save();
                ctx.globalAlpha = Math.max(0.6, props.maskOpacity);
                ctx.drawImage(
                    maskRenderer.canvasElement,
                    drawX,
                    drawY,
                    drawWidth,
                    drawHeight,
                );
                ctx.restore();
            }

            if (!drawing || props.method !== "sam") return;

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
        };
    }, [
        viewport,
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
        if (props.method === "sam") {
            hint =
                props.prompt.length === 0
                    ? (props.promptHint ??
                      "Click the object to segment it · Shift-click or right-click to exclude a region")
                    : `${props.prompt.length} point${props.prompt.length === 1 ? "" : "s"} · keep clicking to refine`;
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
                    onContextMenu={(event) => {
                        if (!drawing) return;
                        event.preventDefault();
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
            </div>
        </div>
    );
}
