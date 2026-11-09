import type { RawRle } from "../types";

export interface OverlayMaskKey {
    payload: RawRle;
    color: string;
    selected: boolean;
}

interface OverlayEntry {
    width: number;
    height: number;
    outlineMode: string;
    masks: OverlayMaskKey[];
    fill: HTMLCanvasElement;
    edge: HTMLCanvasElement;
    bytes: number;
}

/** A bounded cache of composed, frame-resolution mask overlays. */
export class OverlayCache {
    private readonly entries = new Map<number, OverlayEntry>();
    private bytes = 0;
    private readonly maxBytes: number;

    constructor(maxBytes = 64 * 1024 * 1024) {
        this.maxBytes = maxBytes;
    }

    get(
        frame: number,
        width: number,
        height: number,
        outlineMode: string,
        masks: OverlayMaskKey[],
    ): OverlayEntry | undefined {
        const entry = this.entries.get(frame);
        if (!entry) return undefined;
        const matches =
            entry.width === width &&
            entry.height === height &&
            entry.outlineMode === outlineMode &&
            entry.masks.length === masks.length &&
            masks.every((mask, i) => {
                const cached = entry.masks[i];
                return cached.payload === mask.payload &&
                    cached.color === mask.color &&
                    cached.selected === mask.selected;
            });
        if (!matches) {
            this.remove(frame);
            return undefined;
        }
        this.entries.delete(frame);
        this.entries.set(frame, entry);
        return entry;
    }

    set(
        frame: number,
        outlineMode: string,
        masks: OverlayMaskKey[],
        fill: HTMLCanvasElement,
        edge: HTMLCanvasElement,
    ): void {
        this.remove(frame);
        const bytes = fill.width * fill.height * 4 * 2;
        if (bytes > this.maxBytes) return;
        while (this.bytes + bytes > this.maxBytes) {
            const oldest = this.entries.keys().next();
            if (oldest.done) break;
            this.remove(oldest.value);
        }
        const copy = (source: HTMLCanvasElement): HTMLCanvasElement => {
            const canvas = document.createElement("canvas");
            canvas.width = source.width;
            canvas.height = source.height;
            const ctx = canvas.getContext("2d");
            if (!ctx) throw new Error("2D canvas context is unavailable.");
            ctx.drawImage(source, 0, 0);
            return canvas;
        };
        this.entries.set(frame, {
            width: fill.width,
            height: fill.height,
            outlineMode,
            masks: masks.slice(),
            fill: copy(fill),
            edge: copy(edge),
            bytes,
        });
        this.bytes += bytes;
    }

    clear(): void {
        this.entries.clear();
        this.bytes = 0;
    }

    private remove(frame: number): void {
        const entry = this.entries.get(frame);
        if (!entry) return;
        this.entries.delete(frame);
        this.bytes -= entry.bytes;
    }
}
