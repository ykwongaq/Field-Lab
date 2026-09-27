import type { DecodedMask, RawRle } from "../types";
import { API_BASE } from "./apiBase";

/**
 * Client for the backend mask-decoding endpoint. The frontend sends raw RLE
 * masks (size + counts) and receives the decoded foreground runs, so no RLE
 * decoding happens in the browser.
 */
const ENDPOINT = `${API_BASE}/api/decode/masks`;

export async function decodeMasks(masks: RawRle[]): Promise<DecodedMask[]> {
	if (masks.length === 0) return [];

	const response = await fetch(ENDPOINT, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ masks }),
	});
	if (!response.ok) {
		throw new Error(`Mask decode request failed (${response.status}).`);
	}
	const payload = (await response.json()) as { masks: DecodedMask[] };
	if (!Array.isArray(payload.masks) || payload.masks.length !== masks.length) {
		throw new Error("Mask decode response did not match the request.");
	}
	return payload.masks;
}

const DEFAULT_MAX_ENTRIES = 512;

/** A mask that needs to be decoded, together with its cache identity. */
export interface MaskRequest {
	trackletId: number;
	frameIndex: number;
	payload: RawRle;
}

/**
 * Bounded cache of decoded masks, keyed by `trackletId:frameIndex`.
 *
 * `resolveBatch` is the single entry point the UI uses: it returns cached
 * masks immediately, awaits any decode already in flight for the same key,
 * and batches the remaining masks into one network round-trip. The in-flight
 * bookkeeping is synchronous, so concurrent callers (the current-frame render
 * and the look-ahead prefetcher) can never duplicate a request.
 *
 * Every entry remembers the RLE object it was decoded from. The mask tools
 * replace a tracklet's RLE when a mask is edited, so a cached entry is only
 * used while the request carries the very same RLE object; otherwise the
 * new payload is decoded and replaces the stale entry.
 */
export class MaskCache {
	private readonly store = new Map<
		string,
		{ payload: RawRle; mask: DecodedMask }
	>();
	private readonly inflight = new Map<
		string,
		{ payload: RawRle; promise: Promise<DecodedMask> }
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
	 * Resolve a batch of mask requests. Returns one entry per input, in the
	 * same order; masks that cannot be decoded resolve to `undefined`.
	 */
	async resolveBatch(
		requests: MaskRequest[],
	): Promise<(DecodedMask | undefined)[]> {
		const results = new Array<DecodedMask | undefined>(requests.length);
		const waits: Promise<void>[] = [];
		const toFetch: { index: number; request: MaskRequest; key: string }[] = [];

		for (let i = 0; i < requests.length; i++) {
			const request = requests[i];
			const key = MaskCache.key(request.trackletId, request.frameIndex);
			const cached = this.get(key, request.payload);
			if (cached) {
				results[i] = cached;
				continue;
			}
			const existing = this.inflight.get(key);
			if (existing && existing.payload === request.payload) {
				const index = i;
				waits.push(
					existing.promise.then(
						(mask) => {
							results[index] = mask;
						},
						() => {
							results[index] = undefined;
						},
					),
				);
				continue;
			}
			toFetch.push({ index: i, request, key });
		}

		if (toFetch.length > 0) {
			const batch = decodeMasks(toFetch.map((entry) => entry.request.payload));
			toFetch.forEach(({ index, key, request }, i) => {
				const single = batch
					.then((decoded) => decoded[i])
					.then((mask) => {
						this.set(key, request.payload, mask);
						return mask;
					});
				this.inflight.set(key, { payload: request.payload, promise: single });
				waits.push(
					single.then(
						(mask) => {
							results[index] = mask;
						},
						() => {
							results[index] = undefined;
						},
					),
				);
			});
		}

		await Promise.all(waits);

		for (const { key, request } of toFetch) {
			if (this.inflight.get(key)?.payload === request.payload)
				this.inflight.delete(key);
		}

		return results;
	}

	private evict(): void {
		while (this.store.size > this.maxEntries) {
			const oldest = this.store.keys().next().value;
			if (oldest === undefined) break;
			this.store.delete(oldest);
		}
	}
}
