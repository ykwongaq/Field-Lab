import type { ForegroundRun } from "../types";

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
     * The erosion is done by compositing whole canvases rather than per run, so
     * its cost does not grow with the number of runs — a full-frame mask outlines
     * as cheaply as a small one. The canvas is *not* cleared here, so several
     * masks can be outlined into one buffer and blitted together at their own
     * alpha.
     */
    strokeRuns(runs: ForegroundRun[], color: string, width: number): void {
        const w = this.canvas.width;
        const h = this.canvas.height;
        if (w === 0 || h === 0 || runs.length === 0) return;
        const thickness = Math.max(1, Math.round(width));
        const shape = this.ensureShape();
        const shapeCtx = this.shapeCtx!;
        const eroded = this.ensureEroded();
        const erodedCtx = this.erodedCtx!;

        // The mask itself, drawn once, is the source every erosion step samples.
        shapeCtx.clearRect(0, 0, w, h);
        shapeCtx.fillStyle = "#fff";
        for (const run of runs) shapeCtx.fillRect(run.x, run.y, 1, run.length);

        // Erosion = the mask intersected with a copy of itself shifted into each
        // neighbour, so only pixels whose whole neighbourhood is foreground
        // survive. `destination-in` keeps the destination wherever the shifted
        // source is opaque, which is exactly that intersection.
        erodedCtx.save();
        erodedCtx.globalCompositeOperation = "source-over";
        erodedCtx.clearRect(0, 0, w, h);
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
        shapeCtx.fillRect(0, 0, w, h);
        shapeCtx.restore();

        this.ctx.drawImage(shape, 0, 0);
    }

    private ensureShape(): HTMLCanvasElement {
        if (!this.shape) {
            this.shape = document.createElement("canvas");
            this.shape.width = this.canvas.width;
            this.shape.height = this.canvas.height;
            this.shapeCtx = this.shape.getContext("2d");
        }
        return this.shape;
    }

    private ensureEroded(): HTMLCanvasElement {
        if (!this.eroded) {
            this.eroded = document.createElement("canvas");
            this.eroded.width = this.canvas.width;
            this.eroded.height = this.canvas.height;
            this.erodedCtx = this.eroded.getContext("2d");
        }
        return this.eroded;
    }
}
