import type { ForegroundRun, RawRle } from "../types";


export function decodeCounts(counts: string): number[] {
	const result: number[] = [];
	let p = 0;
	while (p < counts.length) {
		let x = 0;
		let k = 0;
		let more = true;
		while (more) {
			const c = counts.charCodeAt(p) - 48;
			x += (c & 0x1f) * 2 ** (5 * k);
			more = (c & 0x20) !== 0;
			p += 1;
			k += 1;
			if (!more && c & 0x10) x -= 2 ** (5 * k);
		}
		if (result.length > 2) x += result[result.length - 2];
		result.push(x);
	}
	return result;
}

export function encodeCounts(counts: number[]): string {
	const chars: number[] = [];
	for (let i = 0; i < counts.length; i++) {
		let x = counts[i];
		if (i > 2) x -= counts[i - 2];
		let more = true;
		while (more) {
			let c = x & 0x1f;
			x >>= 5;
			more = c & 0x10 ? x !== -1 : x !== 0;
			if (more) c |= 0x20;
			chars.push(c + 48);
		}
	}
	let out = "";
	for (let i = 0; i < chars.length; i += 8192) {
		out += String.fromCharCode(...chars.slice(i, i + 8192));
	}
	return out;
}

export function countsOf(rle: RawRle): number[] {
	return typeof rle.counts === "string"
		? decodeCounts(rle.counts)
		: [...rle.counts];
}

export function rleToBitmap(rle: RawRle): Uint8Array {
	const [height, width] = rle.size;
	const total = height * width;
	const bitmap = new Uint8Array(total);
	const counts = countsOf(rle);
	let pos = 0;
	for (let i = 0; i < counts.length && pos < total; i++) {
		const run = Math.min(counts[i], total - pos);
		if (i % 2 === 1) bitmap.fill(1, pos, pos + run);
		pos += run;
	}
	return bitmap;
}

export function bitmapToRle(
	bitmap: Uint8Array,
	height: number,
	width: number,
): RawRle {
	const total = height * width;
	const counts: number[] = [];
	let current = 0;
	let run = 0;
	for (let p = 0; p < total; p++) {
		const value = bitmap[p] ? 1 : 0;
		if (value !== current) {
			counts.push(run);
			run = 0;
			current = value;
		}
		run += 1;
	}
	counts.push(run);
	return { size: [height, width], counts: encodeCounts(counts) };
}

export function rleArea(rle: RawRle): number {
	const counts = countsOf(rle);
	let area = 0;
	for (let i = 1; i < counts.length; i += 2) area += counts[i];
	return area;
}

/** `true` when the mask has no foreground pixel. */
export function rleIsEmpty(rle: RawRle | null | undefined): boolean {
	return !rle || rleArea(rle) === 0;
}


export function bitmapToRuns(
	bitmap: Uint8Array,
	height: number,
	width: number,
): ForegroundRun[] {
	const runs: ForegroundRun[] = [];
	for (let x = 0; x < width; x++) {
		const base = x * height;
		let y = 0;
		while (y < height) {
			if (!bitmap[base + y]) {
				y += 1;
				continue;
			}
			const start = y;
			while (y < height && bitmap[base + y]) y += 1;
			runs.push({ x, y: start, length: y - start });
		}
	}
	return runs;
}

function assertSameSize(a: RawRle, b: RawRle): void {
	if (a.size[0] !== b.size[0] || a.size[1] !== b.size[1]) {
		throw new Error(
			`Mask size mismatch: ${a.size.join("x")} vs ${b.size.join("x")}`,
		);
	}
}

/** `a ∪ b`. `a` may be `null` (then the result is `b`). */
export function unionRle(a: RawRle | null, b: RawRle): RawRle {
	if (!a) return b;
	assertSameSize(a, b);
	const bitmap = rleToBitmap(a);
	const other = rleToBitmap(b);
	for (let p = 0; p < bitmap.length; p++) if (other[p]) bitmap[p] = 1;
	return bitmapToRle(bitmap, a.size[0], a.size[1]);
}

/** `a \ b`; `null` when nothing is left. */
export function subtractRle(a: RawRle, b: RawRle): RawRle | null {
	assertSameSize(a, b);
	const bitmap = rleToBitmap(a);
	const other = rleToBitmap(b);
	let area = 0;
	for (let p = 0; p < bitmap.length; p++) {
		if (other[p]) bitmap[p] = 0;
		area += bitmap[p];
	}
	return area === 0 ? null : bitmapToRle(bitmap, a.size[0], a.size[1]);
}

/** `true` when the two masks share at least one pixel. */
export function rleIntersects(a: RawRle, b: RawRle): boolean {
	assertSameSize(a, b);
	const bitmap = rleToBitmap(a);
	const other = rleToBitmap(b);
	for (let p = 0; p < bitmap.length; p++) if (bitmap[p] && other[p]) return true;
	return false;
}
