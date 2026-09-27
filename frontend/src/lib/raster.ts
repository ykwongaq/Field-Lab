import type { DecodedMask, ForegroundRun, RawRle } from "../types";
import {
	bitmapToRle,
	bitmapToRuns,
	rleArea,
	rleToBitmap,
	subtractRle,
	unionRle,
} from "./rle";

export interface FramePoint {
	x: number;
	y: number;
}

export type PaintMode = "add" | "erase";

const COVERAGE_THRESHOLD = 128;

/**
 * Offscreen canvas at frame resolution used to rasterise polygons and brush
 * strokes.
 */
export class RasterCanvas {
	readonly width: number;
	readonly height: number;
	private readonly canvas: HTMLCanvasElement;
	private readonly ctx: CanvasRenderingContext2D;

	constructor(width: number, height: number) {
		this.width = width;
		this.height = height;
		this.canvas = document.createElement("canvas");
		this.canvas.width = width;
		this.canvas.height = height;
		const ctx = this.canvas.getContext("2d", { willReadFrequently: true });
		if (!ctx) throw new Error("2D canvas context is unavailable.");
		this.ctx = ctx;
		this.ctx.fillStyle = "#fff";
		this.ctx.strokeStyle = "#fff";
		this.ctx.lineCap = "round";
		this.ctx.lineJoin = "round";
	}

	clear(): void {
		this.ctx.clearRect(0, 0, this.width, this.height);
	}

	/** Fill a closed polygon */
	fillPolygon(points: FramePoint[]): void {
		if (points.length < 3) return;
		this.ctx.beginPath();
		this.ctx.moveTo(points[0].x, points[0].y);
		for (let i = 1; i < points.length; i++)
			this.ctx.lineTo(points[i].x, points[i].y);
		this.ctx.closePath();
		this.ctx.fill("evenodd");
	}

	strokeSegment(from: FramePoint, to: FramePoint, size: number): void {
		this.ctx.lineWidth = Math.max(1, size);
		this.ctx.beginPath();
		this.ctx.moveTo(from.x, from.y);
        
		const dx = to.x - from.x;
		const dy = to.y - from.y;
		if (dx === 0 && dy === 0) this.ctx.lineTo(to.x + 0.01, to.y);
		else this.ctx.lineTo(to.x, to.y);
		this.ctx.stroke();
	}


    toBitmap(): Uint8Array {
		const { width, height } = this;
		const rgba = this.ctx.getImageData(0, 0, width, height).data;
		const bitmap = new Uint8Array(width * height);
		for (let y = 0; y < height; y++) {
			const row = y * width;
			for (let x = 0; x < width; x++) {
				if (rgba[(row + x) * 4 + 3] >= COVERAGE_THRESHOLD)
					bitmap[x * height + y] = 1;
			}
		}
		return bitmap;
	}


    toRle(): RawRle | null {
		const rle = bitmapToRle(this.toBitmap(), this.height, this.width);
		return rleArea(rle) === 0 ? null : rle;
	}
}

export function polygonToRle(
	points: FramePoint[],
	width: number,
	height: number,
): RawRle | null {
	const raster = new RasterCanvas(width, height);
	raster.fillPolygon(points);
	return raster.toRle();
}


export function composeRle(
	base: RawRle | null,
	patch: RawRle | null,
	mode: PaintMode,
): RawRle | null {
	if (!patch) return base;
	if (mode === "add") return unionRle(base, patch);
	if (!base) return null;
	return subtractRle(base, patch);
}

export function rleToDecoded(rle: RawRle): DecodedMask {
	const [height, width] = rle.size;
	return {
		height,
		width,
		runs: bitmapToRuns(rleToBitmap(rle), height, width),
	};
}

export function runsContain(
	runs: ForegroundRun[],
	x: number,
	y: number,
): boolean {
	const column = Math.floor(x);
	const row = Math.floor(y);
	for (const run of runs) {
		if (run.x !== column) continue;
		if (row >= run.y && row < run.y + run.length) return true;
	}
	return false;
}

/** Foreground pixel count of decoded runs. */
export function runsArea(runs: ForegroundRun[]): number {
	let area = 0;
	for (const run of runs) area += run.length;
	return area;
}
