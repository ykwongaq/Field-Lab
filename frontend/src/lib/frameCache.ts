import type { FrameSource } from "./frames";

const MAX_ENTRIES = 40;

/**
 * LRU cache of decoded frame bitmaps. JPEG frames decode to full-resolution
 * bitmaps, so the cache is intentionally small and evicts the oldest frames.
 *
 * Frames are fetched through a `FrameSource`, which is what keeps a session's
 * frames zero-effort to cache: they are immutable once the session is built.
 */
export class FrameCache {
    private readonly cache = new Map<number, ImageBitmap>();
    private readonly order: number[] = [];
    private readonly inflight = new Map<number, Promise<ImageBitmap>>();
    private readonly source: FrameSource;

    constructor(source: FrameSource) {
        this.source = source;
    }

    async get(index: number): Promise<ImageBitmap> {
        const cached = this.cache.get(index);
        if (cached) {
            this.touch(index);
            return cached;
        }

        const pending = this.inflight.get(index);
        if (pending) return pending;

        const promise = this.load(index);
        this.inflight.set(index, promise);
        try {
            return await promise;
        } finally {
            this.inflight.delete(index);
        }
    }

    private async load(index: number): Promise<ImageBitmap> {
        const bitmap = await createImageBitmap(await this.source.frame(index));
        this.cache.set(index, bitmap);
        this.order.push(index);
        this.evict();
        return bitmap;
    }

    preload(indices: number[]): void {
        for (const index of indices) {
            void this.get(index).catch(() => {
                /* a missing frame is tolerated and reported by the caller */
            });
        }
    }

    private touch(index: number): void {
        const position = this.order.indexOf(index);
        if (position !== -1) this.order.splice(position, 1);
        this.order.push(index);
    }

    private evict(): void {
        while (this.order.length > MAX_ENTRIES) {
            const victim = this.order.shift();
            if (victim === undefined) break;
            const bitmap = this.cache.get(victim);
            if (bitmap) bitmap.close();
            this.cache.delete(victim);
        }
    }
}
