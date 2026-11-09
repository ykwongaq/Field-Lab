import type { ForegroundRun } from "../types";
import type { FramePoint } from "./raster";

/**
 * Boundary stroke width, as a fraction of the frame's shortest side.
 *
 * The outline is measured in *frame* pixels, not screen pixels, so it thickens
 * with the mask when the picture is zoomed in and reads the same at every
 * preview size. 0.2% of the shortest side is ~2.2 px on a 1080p frame — thick
 * enough to stay legible over a busy background, thin enough not to swallow a
 * small mask.
 */
const BOUNDARY_FRACTION = 0.002;
/** Floor for tiny frames, so the outline never disappears. */
const MIN_BOUNDARY_PX = 1;
/** Ceiling so a large frame does not get a fat, cartoonish outline. */
const MAX_BOUNDARY_PX = 8;

/** Frame-pixel stroke width for a mask boundary at this frame size. */
export function boundaryWidthFor(width: number, height: number): number {
    const shortest = Math.min(width, height);
    if (!(shortest > 0)) return MIN_BOUNDARY_PX;
    return Math.min(
        MAX_BOUNDARY_PX,
        Math.max(MIN_BOUNDARY_PX, shortest * BOUNDARY_FRACTION),
    );
}

/**
 * A closed path through every ring, ready to stroke or fill.
 *
 * Each ring is closed, so a dash pattern wraps around the whole loop instead of
 * stopping short at the vertex it started from; holes become separate subpaths
 * and read as holes under an even-odd fill.
 */
export function ringsToPath(rings: FramePoint[][]): Path2D {
    const path = new Path2D();
    for (const ring of rings) {
        if (ring.length < 3) continue;
        path.moveTo(ring[0].x, ring[0].y);
        for (let i = 1; i < ring.length; i++) {
            path.lineTo(ring[i].x, ring[i].y);
        }
        path.closePath();
    }
    return path;
}

/**
 * The eight neighbour directions an outline is eroded along.
 *
 * They are scaled by a whole number of pixels in `strokeRuns`: a fractional
 * offset would antialias the shifted copy, and a partial-alpha source under
 * `destination-in` erodes the mask only partway, leaving a ragged, half-faded
 * rim instead of a clean band.
 */
const EROSION_DIRECTIONS: ReadonlyArray<readonly [number, number]> = [
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1],
    [1, 1],
    [1, -1],
    [-1, 1],
    [-1, -1],
];

/**
 * Renders decoded RLE masks onto an offscreen canvas at full frame resolution.
 * Foreground runs are drawn as 1px-wide vertical strips, which mirrors the
 * column-major encoding and avoids materialising the full bitmap.
 *
 * The same canvas also draws mask *boundaries* (`strokeRuns`), so a caller can
 * keep the translucent fill and the opaque outline on separate buffers and blit
 * each at its own alpha.
 */
export class MaskRenderer {
    private readonly canvas: HTMLCanvasElement;
    private readonly ctx: CanvasRenderingContext2D;
    // Scratch buffers, built lazily, used only while outlining (see `strokeRuns`).
    private shape: HTMLCanvasElement | null = null;
    private shapeCtx: CanvasRenderingContext2D | null = null;
    private eroded: HTMLCanvasElement | null = null;
    private erodedCtx: CanvasRenderingContext2D | null = null;

    constructor(width: number, height: number) {
        this.canvas = document.createElement("canvas");
        this.canvas.width = width;
        this.canvas.height = height;
        const ctx = this.canvas.getContext("2d");
        if (!ctx) throw new Error("2D canvas context is unavailable.");
        this.ctx = ctx;
    }

    get canvasElement(): HTMLCanvasElement {
        return this.canvas;
    }

    clear(): void {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }

    drawRuns(runs: ForegroundRun[], color: string): void {
        this.ctx.fillStyle = color;
        for (const run of runs) {
            this.ctx.fillRect(run.x, run.y, 1, run.length);
        }
    }

    /**
     * Outline `runs`: the rim left when the mask is eroded by `width` frame
     * pixels, drawn in `color` onto this renderer's canvas. The rim hugs the
     * inside of the mask.
     *
     * Scratch canvases cover only the mask bounds and the erosion padding.
     * The output canvas is not cleared, so multiple outlines can be combined.
     */
    strokeRuns(runs: ForegroundRun[], color: string, width: number): void {
        const w = this.canvas.width;
        const h = this.canvas.height;
        if (w === 0 || h === 0 || runs.length === 0) return;
        const thickness = Math.max(1, Math.round(width));
        let minX = w;
        let minY = h;
        let maxX = 0;
        let maxY = 0;
        for (const run of runs) {
            minX = Math.min(minX, run.x);
            minY = Math.min(minY, run.y);
            maxX = Math.max(maxX, run.x + 1);
            maxY = Math.max(maxY, run.y + run.length);
        }
        const left = Math.max(0, minX - thickness);
        const top = Math.max(0, minY - thickness);
        const right = Math.min(w, maxX + thickness);
        const bottom = Math.min(h, maxY + thickness);
        const scratchWidth = right - left;
        const scratchHeight = bottom - top;
        if (scratchWidth <= 0 || scratchHeight <= 0) return;
        const shape = this.ensureShape(scratchWidth, scratchHeight);
        const shapeCtx = this.shapeCtx!;
        const eroded = this.ensureEroded(scratchWidth, scratchHeight);
        const erodedCtx = this.erodedCtx!;

        // The mask itself, drawn once, is the source every erosion step samples.
        shapeCtx.clearRect(0, 0, scratchWidth, scratchHeight);
        shapeCtx.fillStyle = "#fff";
        for (const run of runs)
            shapeCtx.fillRect(run.x - left, run.y - top, 1, run.length);

        // Erosion = the mask intersected with a copy of itself shifted into each
        // neighbour, so only pixels whose whole neighbourhood is foreground
        // survive. `destination-in` keeps the destination wherever the shifted
        // source is opaque, which is exactly that intersection.
        erodedCtx.save();
        erodedCtx.globalCompositeOperation = "source-over";
        erodedCtx.clearRect(0, 0, scratchWidth, scratchHeight);
        erodedCtx.drawImage(shape, 0, 0);
        erodedCtx.globalCompositeOperation = "destination-in";
        for (const [dx, dy] of EROSION_DIRECTIONS) {
            erodedCtx.drawImage(shape, dx * thickness, dy * thickness);
        }
        erodedCtx.restore();

        // Rim = mask − erosion. Punching the erosion out of the mask leaves a
        // band `thickness` wide along the boundary; recolour it in one fill.
        shapeCtx.save();
        shapeCtx.globalCompositeOperation = "destination-out";
        shapeCtx.drawImage(eroded, 0, 0);
        shapeCtx.globalCompositeOperation = "source-in";
        shapeCtx.fillStyle = color;
        shapeCtx.fillRect(0, 0, scratchWidth, scratchHeight);
        shapeCtx.restore();

        this.ctx.drawImage(shape, left, top);
    }

    private ensureShape(width: number, height: number): HTMLCanvasElement {
        if (!this.shape) {
            this.shape = document.createElement("canvas");
            this.shapeCtx = this.shape.getContext("2d");
            if (!this.shapeCtx) throw new Error("2D canvas context is unavailable.");
        }
        if (this.shape.width !== width) this.shape.width = width;
        if (this.shape.height !== height) this.shape.height = height;
        return this.shape;
    }

    private ensureEroded(width: number, height: number): HTMLCanvasElement {
        if (!this.eroded) {
            this.eroded = document.createElement("canvas");
            this.erodedCtx = this.eroded.getContext("2d");
            if (!this.erodedCtx) throw new Error("2D canvas context is unavailable.");
        }
        if (this.eroded.width !== width) this.eroded.width = width;
        if (this.eroded.height !== height) this.eroded.height = height;
        return this.eroded;
    }
}
