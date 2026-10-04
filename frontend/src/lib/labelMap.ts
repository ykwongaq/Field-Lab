import type { RawRle } from "../types";
import { countsOf, encodeCounts } from "./rle";
import { crc32 } from "./zipWriter";



export const MAX_LABEL_MAP_CLASS_ID = 255;

export function labelMapEntryFor(frameName: string): string {
	const base = frameName.split("/").pop() ?? frameName;
	return `masks/${base.replace(/\.[^.]+$/, "")}.png`;
}


export async function decodeLabelMapPng(
	blob: Blob,
	width: number,
	height: number,
): Promise<Uint8Array> {
	const bitmap = await createImageBitmap(blob, {
		premultiplyAlpha: "none",
		colorSpaceConversion: "none",
	});
	try {
		if (bitmap.width !== width || bitmap.height !== height) {
			throw new Error(
				`Label map is ${bitmap.width}x${bitmap.height}, expected ${width}x${height}`,
			);
		}
		const canvas = new OffscreenCanvas(width, height);
		const ctx = canvas.getContext("2d", { willReadFrequently: true });
		if (!ctx) throw new Error("2D canvas context unavailable");
		ctx.drawImage(bitmap, 0, 0);
		const rgba = ctx.getImageData(0, 0, width, height).data;
		const ids = new Uint8Array(width * height);
		for (let p = 0, q = 0; p < ids.length; p++, q += 4) ids[p] = rgba[q];
		return ids;
	} finally {
		bitmap.close();
	}
}

const PNG_SIGNATURE = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);

async function zlibDeflate(bytes: Uint8Array): Promise<Uint8Array> {
	const stream = new Blob([bytes as BlobPart])
		.stream()
		.pipeThrough(new CompressionStream("deflate"));
	return new Uint8Array(await new Response(stream).arrayBuffer());
}

function pngChunk(type: string, data: Uint8Array): Uint8Array {
	const out = new Uint8Array(12 + data.length);
	const view = new DataView(out.buffer);
	view.setUint32(0, data.length);
	for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
	out.set(data, 8);
	view.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
	return out;
}


export async function encodeLabelMapPng(
	ids: Uint8Array,
	width: number,
	height: number,
): Promise<Blob> {
	if (ids.length !== width * height) {
		throw new Error(`Label map has ${ids.length} pixels, expected ${width * height}`);
	}
	const header = new Uint8Array(13);
	const view = new DataView(header.buffer);
	view.setUint32(0, width);
	view.setUint32(4, height);
	header[8] = 8; // bit depth
	header[9] = 0; // colour type: greyscale
	header[10] = 0; // compression
	header[11] = 0; // filter method
	header[12] = 0; // no interlace

	const raw = new Uint8Array((width + 1) * height);
	for (let y = 0; y < height; y++) {
		raw[y * (width + 1)] = 0;
		raw.set(ids.subarray(y * width, (y + 1) * width), y * (width + 1) + 1);
	}
	const idat = await zlibDeflate(raw);
	return new Blob(
		[
			PNG_SIGNATURE,
			pngChunk("IHDR", header),
			pngChunk("IDAT", idat),
			pngChunk("IEND", new Uint8Array(0)),
		] as BlobPart[],
		{ type: "image/png" },
	);
}


export function labelMapToClassMasks(
	ids: Uint8Array,
	width: number,
	height: number,
): Map<number, RawRle> {
	const toggles = new Map<number, number[]>();
	const toggle = (id: number, position: number) => {
		if (id === 0) return;
		let list = toggles.get(id);
		if (!list) {
			list = [];
			toggles.set(id, list);
		}
		list.push(position);
	};

	const total = width * height;
	let previous = 0;
	let position = 0;
	for (let x = 0; x < width; x++) {
		for (let y = 0; y < height; y++, position++) {
			const id = ids[y * width + x];
			if (id !== previous) {
				toggle(previous, position);
				toggle(id, position);
				previous = id;
			}
		}
	}
	toggle(previous, total);

	const masks = new Map<number, RawRle>();
	for (const [id, positions] of toggles) {
		const counts: number[] = [];
		let last = 0;
		for (const p of positions) {
			counts.push(p - last);
			last = p;
		}
		if (last < total) counts.push(total - last);
		masks.set(id, { size: [height, width], counts: encodeCounts(counts) });
	}
	return masks;
}


export function classMasksToLabelMap(
	masks: Iterable<{ id: number; rle: RawRle }>,
	width: number,
	height: number,
): Uint8Array {
	const ids = new Uint8Array(width * height);
	for (const { id, rle } of masks) {
		if (id <= 0 || id > MAX_LABEL_MAP_CLASS_ID) {
			throw new Error(
				`Category id ${id} cannot be stored in an 8-bit label map (1..${MAX_LABEL_MAP_CLASS_ID}).`,
			);
		}
		if (rle.size[0] !== height || rle.size[1] !== width) {
			throw new Error(
				`Mask of class ${id} is ${rle.size.join("x")}, expected ${height}x${width}`,
			);
		}
		const counts = countsOf(rle);
		let position = 0;
		for (let i = 0; i < counts.length; i++) {
			const run = counts[i];
			if (i % 2 === 1) {
				for (let p = position; p < position + run; p++) {
					const x = Math.floor(p / height);
					const y = p - x * height;
					ids[y * width + x] = id;
				}
			}
			position += run;
		}
	}
	return ids;
}
