import assert from "node:assert/strict";
import { test } from "node:test";
import {
    bitmapToRle,
    bitmapToRuns,
    countsOf,
    rleToBitmap,
    rleToRuns,
} from "../src/lib/rle.ts";
import { MaskRenderer } from "../src/lib/mask.ts";
import { OverlayCache } from "../src/lib/overlayCache.ts";

class Canvas {
    _width = 300;
    _height = 150;
    pixels = new Uint8Array(this._width * this._height);
    ctx = new Context(this);
    get width() { return this._width; }
    set width(value) { this._width = value; this.reset(); }
    get height() { return this._height; }
    set height(value) { this._height = value; this.reset(); }
    reset() {
        this.pixels = new Uint8Array(this._width * this._height);
        this.ctx.globalCompositeOperation = "source-over";
    }
    getContext() { return this.ctx; }
}

class Context {
    globalCompositeOperation = "source-over";
    stack = [];
    constructor(canvas) { this.canvas = canvas; }
    save() { this.stack.push(this.globalCompositeOperation); }
    restore() { this.globalCompositeOperation = this.stack.pop(); }
    clearRect(x, y, w, h) {
        for (let py = Math.max(y, 0); py < Math.min(y + h, this.canvas.height); py++)
            for (let px = Math.max(x, 0); px < Math.min(x + w, this.canvas.width); px++)
                this.canvas.pixels[py * this.canvas.width + px] = 0;
    }
    composite(source) {
        for (let i = 0; i < this.canvas.pixels.length; i++) {
            const dst = this.canvas.pixels[i];
            const src = source[i];
            switch (this.globalCompositeOperation) {
                case "source-over": this.canvas.pixels[i] = src || dst; break;
                case "destination-in": this.canvas.pixels[i] = dst && src; break;
                case "destination-out": this.canvas.pixels[i] = dst && !src; break;
                case "source-in": this.canvas.pixels[i] = src && dst; break;
                default: throw new Error("Unsupported composite operation");
            }
        }
    }
    fillRect(x, y, w, h) {
        const source = new Uint8Array(this.canvas.pixels.length);
        for (let py = Math.max(y, 0); py < Math.min(y + h, this.canvas.height); py++)
            for (let px = Math.max(x, 0); px < Math.min(x + w, this.canvas.width); px++)
                source[py * this.canvas.width + px] = 1;
        this.composite(source);
    }
    drawImage(image, x, y) {
        const source = new Uint8Array(this.canvas.pixels.length);
        for (let py = 0; py < image.height; py++)
            for (let px = 0; px < image.width; px++) {
                const dx = x + px;
                const dy = y + py;
                if (dx >= 0 && dy >= 0 && dx < this.canvas.width && dy < this.canvas.height)
                    source[dy * this.canvas.width + dx] = image.pixels[py * image.width + px];
            }
        this.composite(source);
    }
}

globalThis.document = { createElement: () => new Canvas() };

function referenceOutline(runs, width, height, thickness) {
    const shape = new Canvas();
    const eroded = new Canvas();
    shape.width = eroded.width = width;
    shape.height = eroded.height = height;
    const a = shape.getContext("2d");
    const b = eroded.getContext("2d");
    for (const run of runs) a.fillRect(run.x, run.y, 1, run.length);
    b.drawImage(shape, 0, 0);
    b.globalCompositeOperation = "destination-in";
    for (const [x, y] of [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]])
        b.drawImage(shape, x * thickness, y * thickness);
    a.globalCompositeOperation = "destination-out";
    a.drawImage(eroded, 0, 0);
    return shape.pixels;
}

test("direct RLE decoding matches bitmap decoding, including column crossings", () => {
    let seed = 123456;
    const random = () => (seed = (seed * 1664525 + 1013904223) >>> 0);
    for (let n = 0; n < 300; n++) {
        const width = 1 + random() % 30;
        const height = 1 + random() % 30;
        const bitmap = Uint8Array.from({ length: width * height }, () => (random() >>> 28) < 6 ? 1 : 0);
        const rle = bitmapToRle(bitmap, height, width);
        for (const counts of [rle.counts, countsOf(rle)]) {
            const input = { size: rle.size, counts };
            assert.deepEqual(rleToRuns(input), bitmapToRuns(rleToBitmap(input), height, width));
        }
    }
    const zeroRuns = { size: [4, 2], counts: [0, 2, 0, 4, 0, 2] };
    assert.deepEqual(rleToRuns(zeroRuns), bitmapToRuns(rleToBitmap(zeroRuns), 4, 2));
});

test("bounded outline matches full-frame erosion for edge and sparse masks", () => {
    let seed = 1;
    const random = () => (seed = (seed * 1664525 + 1013904223) >>> 0);
    for (let n = 0; n < 120; n++) {
        const width = 4 + random() % 24;
        const height = 4 + random() % 24;
        const thickness = 1 + random() % 5;
        const bitmap = new Uint8Array(width * height);
        const left = random() % width;
        const top = random() % height;
        const right = left + 1 + random() % (width - left);
        const bottom = top + 1 + random() % (height - top);
        for (let x = left; x < right; x++)
            for (let y = top; y < bottom; y++)
                bitmap[x * height + y] = (random() >>> 29) ? 1 : 0;
        const runs = bitmapToRuns(bitmap, height, width);
        const renderer = new MaskRenderer(width, height);
        renderer.strokeRuns(runs, "#fff", thickness);
        assert.deepEqual(renderer.canvasElement.pixels, referenceOutline(runs, width, height, thickness));
    }
});

test("overlay cache detects changes, copies canvases, and evicts least recently used frames", () => {
    const cache = new OverlayCache(64);
    const fill = new Canvas();
    const edge = new Canvas();
    fill.width = edge.width = 2;
    fill.height = edge.height = 2;
    fill.pixels[0] = 1;
    const payload = { size: [2, 2], counts: [4] };
    const key = [{ payload, color: "red", selected: false }];
    cache.set(1, "review", key, fill, edge);
    cache.set(2, "review", key, fill, edge);
    fill.pixels[0] = 0;
    assert.equal(cache.get(1, 2, 2, "review", key).fill.pixels[0], 1);
    cache.set(3, "review", key, fill, edge);
    assert.equal(cache.get(2, 2, 2, "review", key), undefined);
    assert.equal(cache.get(1, 2, 2, "review", [{ ...key[0], selected: true }]), undefined);
    assert.equal(cache.get(3, 2, 2, "review", [{ ...key[0], payload: { ...payload } }]), undefined);
    cache.set(4, "review", key, fill, edge);
    assert.equal(cache.get(4, 2, 2, "other", key), undefined);
    cache.set(5, "review", key, fill, edge);
    assert.equal(cache.get(5, 3, 2, "review", key), undefined);
});
