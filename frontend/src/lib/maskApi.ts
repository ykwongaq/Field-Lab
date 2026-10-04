import type { DecodedMask, RawRle } from "../types";
import { rleToDecoded } from "./raster";

/**
 * Decoded-mask cache for the canvas.
 *
 * Decoding happens in the browser (`lib/rle.ts`), so there is no round trip and
 * no batching to schedule: a mask is turned into drawable runs the moment it is
 * asked for. The cache still exists because the display asks for the same masks
 * repeatedly — the current frame, the look-ahead prefetcher, and the mask list
 * all hit the same tracklet/frame pairs.
 *
 * Every entry remembers the RLE object it was decoded from. Editing a mask
 * replaces that object, so a cached entry is only reused while the request
 * carries the very same RLE; otherwise the new payload is decoded and replaces
 * the stale entry.
 */

export interface MaskRequest {
    trackletId: number;
    frameIndex: number;
    payload: RawRle;
}

const DEFAULT_MAX_ENTRIES = 512;

/** Decode one mask, or `undefined` when the RLE is malformed. */
export function decodeMask(payload: RawRle): DecodedMask | undefined {
    try {
        return rleToDecoded(payload);
    } catch {
        return undefined;
    }
}

export class MaskCache {
    private readonly store = new Map<
        string,
        { payload: RawRle; mask: DecodedMask }
    >();
    private readonly maxEntries: number;

    constructor(maxEntries: number = DEFAULT_MAX_ENTRIES) {
        this.maxEntries = maxEntries;
    }

    static key(trackletId: number, frameIndex: number): string {
        return `${trackletId}:${frameIndex}`;
    }

    /** Cached mask for `key`, if it was decoded from exactly this `payload`. */
    get(key: string, payload: RawRle): DecodedMask | undefined {
        const entry = this.store.get(key);
        return entry && entry.payload === payload ? entry.mask : undefined;
    }

    set(key: string, payload: RawRle, mask: DecodedMask): void {
        this.store.set(key, { payload, mask });
        this.evict();
    }

    /**
     * Resolve a batch of mask requests, one entry per input in the same order.
     *
     * Asynchronous only to match the call sites, which used to await a network
     * round trip; masks that cannot be decoded resolve to `undefined`.
     */
    async resolveBatch(
        requests: MaskRequest[],
    ): Promise<(DecodedMask | undefined)[]> {
        return requests.map((request) => {
            const key = MaskCache.key(request.trackletId, request.frameIndex);
            const cached = this.get(key, request.payload);
            if (cached) return cached;
            const mask = decodeMask(request.payload);
            if (mask) this.set(key, request.payload, mask);
            return mask;
        });
    }

    /** Drop everything (used when the clip is replaced). */
    clear(): void {
        this.store.clear();
    }

    private evict(): void {
        while (this.store.size > this.maxEntries) {
            const oldest = this.store.keys().next();
            if (oldest.done) return;
            this.store.delete(oldest.value);
        }
    }
}
