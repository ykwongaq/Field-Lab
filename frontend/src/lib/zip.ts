const EOCD_SIGNATURE = 0x06054b50;
const CD_SIGNATURE = 0x02014b50;
const LFH_SIGNATURE = 0x04034b50;

const ZIP64_EXTRA_ID = 0x0001;
const ZIP64_EOCD_SIGNATURE = 0x06064b50;
const ZIP64_LOCATOR_SIGNATURE = 0x07064b50;
/** Saturation values: a field holding these means "read the Zip64 record". */
const U32_MAX = 0xffffffff;
const U16_MAX = 0xffff;
/** End-of-directory (22) + longest possible comment + the Zip64 locator (20). */
const EOCD_WINDOW_LENGTH = 22 + 65535 + 20;

interface InternalEntry {
    name: string;
    method: number;
    crc32: number;
    compressedSize: number;
    uncompressedSize: number;
    localOffset: number;
    /** Resolved from the local file header on first read of the entry. */
    dataOffset?: number;
}

export interface ZipRawEntry {
    name: string;
    method: number;
    crc32: number;
    compressedSize: number;
    uncompressedSize: number;
    /** Stored bytes as they are in the archive; a `Blob` when sliced lazily. */
    data: Uint8Array | Blob;
}

export class ZipArchive {
    /** The original file; entries are sliced out of it, never read wholesale. */
    private readonly source: Blob;
    private readonly entries: Map<string, InternalEntry>;

    private constructor(source: Blob, entries: Map<string, InternalEntry>) {
        this.source = source;
        this.entries = entries;
    }

    /**
     * Read the archive's directory, keeping the file for later slicing.
     *
     * Only the tail (end-of-directory, plus the Zip64 records when present) and
     * the central directory are read here, so a project holding a multi-gigabyte
     * video opens without being loaded into memory. Entry data is sliced on
     * demand by `readAsBlob`.
     */
    static async fromFile(file: File): Promise<ZipArchive> {
        const tailLength = Math.min(file.size, EOCD_WINDOW_LENGTH);
        const tailStart = file.size - tailLength;
        const tail = new DataView(await file.slice(tailStart).arrayBuffer());
        const eocdInTail = findEocdOffset(tail);
        const eocdPosition = tailStart + eocdInTail;

        let entryCount = tail.getUint16(eocdInTail + 10, true);
        let cdSize = tail.getUint32(eocdInTail + 12, true);
        let cdOffset = tail.getUint32(eocdInTail + 16, true);
        if (
            entryCount === U16_MAX ||
            cdSize === U32_MAX ||
            cdOffset === U32_MAX
        ) {
            const zipped = await readZip64Directory(file, eocdPosition);
            entryCount = zipped.entryCount;
            cdSize = zipped.cdSize;
            cdOffset = zipped.cdOffset;
        }

        const directory = new DataView(
            await file.slice(cdOffset, cdOffset + cdSize).arrayBuffer(),
        );
        const decoder = new TextDecoder("utf-8");
        const entries = new Map<string, InternalEntry>();
        let p = 0;

        for (let i = 0; i < entryCount; i++) {
            if (p + 46 > directory.byteLength) break;
            if (directory.getUint32(p, true) !== CD_SIGNATURE) break;
            const method = directory.getUint16(p + 10, true);
            const crc32 = directory.getUint32(p + 16, true);
            let compressedSize = directory.getUint32(p + 20, true);
            let uncompressedSize = directory.getUint32(p + 24, true);
            const nameLength = directory.getUint16(p + 28, true);
            const extraLength = directory.getUint16(p + 30, true);
            const commentLength = directory.getUint16(p + 32, true);
            let localOffset = directory.getUint32(p + 42, true);
            const name = decoder.decode(
                new Uint8Array(directory.buffer, p + 46, nameLength),
            );

            if (
                compressedSize === U32_MAX ||
                uncompressedSize === U32_MAX ||
                localOffset === U32_MAX
            ) {
                const widened = readZip64Extra(
                    directory,
                    p + 46 + nameLength,
                    extraLength,
                    { compressedSize, uncompressedSize, localOffset },
                );
                compressedSize = widened.compressedSize;
                uncompressedSize = widened.uncompressedSize;
                localOffset = widened.localOffset;
            }

            entries.set(name, {
                name,
                method,
                crc32,
                compressedSize,
                uncompressedSize,
                localOffset,
            });
            p += 46 + nameLength + extraLength + commentLength;
        }

        return new ZipArchive(file, entries);
    }

    getEntries(): string[] {
        return [...this.entries.keys()];
    }

    hasEntry(name: string): boolean {
        return this.entries.has(name);
    }

    async readAsBlob(name: string): Promise<Blob> {
        const entry = this.entries.get(name);
        if (!entry) throw new Error(`Missing zip entry: ${name}`);
        const data = await this.sliceEntry(entry);

        if (entry.method === 0) return data;
        if (entry.method === 8) {
            const raw = new Uint8Array(await data.arrayBuffer());
            return new Blob([await inflateRaw(raw)]);
        }
        if (entry.method === 14) {
            throw new Error(
                `${name}: LZMA compression is not supported by the browser ZIP reader. ` +
                    "Re-export the project with deflated compression " +
                    "(make_projects.py --compression deflated).",
            );
        }
        throw new Error(
            `Unsupported compression method ${entry.method} for ${name}`,
        );
    }

    async readAsText(name: string): Promise<string> {
        return (await this.readAsBlob(name)).text();
    }

    /**
     * The entry's bytes exactly as stored — still compressed for method 8 — so
     * `ZipWriter.addRaw` can copy an entry between archives without inflating it.
     */
    async rawEntry(name: string): Promise<ZipRawEntry> {
        const entry = this.entries.get(name);
        if (!entry) throw new Error(`Missing zip entry: ${name}`);
        return {
            name: entry.name,
            method: entry.method,
            crc32: entry.crc32,
            compressedSize: entry.compressedSize,
            uncompressedSize: entry.uncompressedSize,
            data: await this.sliceEntry(entry),
        };
    }

    private async sliceEntry(entry: InternalEntry): Promise<Blob> {
        const offset = await this.dataOffsetOf(entry);
        return this.source.slice(offset, offset + entry.compressedSize);
    }

    /** The local file header tells us how far past it the data starts. */
    private async dataOffsetOf(entry: InternalEntry): Promise<number> {
        let offset = entry.dataOffset;
        if (offset === undefined) {
            const header = new DataView(
                await this.source
                    .slice(entry.localOffset, entry.localOffset + 30)
                    .arrayBuffer(),
            );
            if (
                header.byteLength < 30 ||
                header.getUint32(0, true) !== LFH_SIGNATURE
            ) {
                throw new Error(
                    `Corrupt ZIP local file header for ${entry.name}`,
                );
            }
            const nameLength = header.getUint16(26, true);
            const extraLength = header.getUint16(28, true);
            offset = entry.localOffset + 30 + nameLength + extraLength;
            entry.dataOffset = offset;
        }
        return offset;
    }
}

function findEocdOffset(view: DataView): number {
    const min = Math.max(0, view.byteLength - 22 - 65535);
    for (let p = view.byteLength - 22; p >= min; p--) {
        if (view.getUint32(p, true) === EOCD_SIGNATURE) return p;
    }
    throw new Error(
        "Not a valid ZIP archive (end-of-central-directory not found).",
    );
}

/** Follow the Zip64 locator to the Zip64 end-of-directory record. */
async function readZip64Directory(
    source: Blob,
    eocdPosition: number,
): Promise<{ entryCount: number; cdSize: number; cdOffset: number }> {
    const locatorPosition = eocdPosition - 20;
    if (locatorPosition < 0) {
        throw new Error(
            "Corrupt ZIP64 archive (missing end-of-directory locator).",
        );
    }
    const locator = new DataView(
        await source.slice(locatorPosition, locatorPosition + 20).arrayBuffer(),
    );
    if (locator.getUint32(0, true) !== ZIP64_LOCATOR_SIGNATURE) {
        throw new Error(
            "Corrupt ZIP64 archive (bad end-of-directory locator).",
        );
    }
    const zippedPosition = Number(locator.getBigUint64(8, true));
    const record = new DataView(
        await source.slice(zippedPosition, zippedPosition + 56).arrayBuffer(),
    );
    if (record.getUint32(0, true) !== ZIP64_EOCD_SIGNATURE) {
        throw new Error("Corrupt ZIP64 archive (bad end-of-directory record).");
    }
    return {
        entryCount: Number(record.getBigUint64(32, true)),
        cdSize: Number(record.getBigUint64(40, true)),
        cdOffset: Number(record.getBigUint64(48, true)),
    };
}

/**
 * Widen the saturated fields of a central-directory entry from its Zip64 extra
 * field, which lists only the values that did not fit, in a fixed order.
 */
function readZip64Extra(
    view: DataView,
    extraStart: number,
    extraLength: number,
    current: {
        compressedSize: number;
        uncompressedSize: number;
        localOffset: number;
    },
): { compressedSize: number; uncompressedSize: number; localOffset: number } {
    const end = extraStart + extraLength;
    for (let p = extraStart; p + 4 <= end; ) {
        const id = view.getUint16(p, true);
        const size = view.getUint16(p + 2, true);
        if (id === ZIP64_EXTRA_ID) {
            const widened = { ...current };
            let q = p + 4;
            if (widened.uncompressedSize === U32_MAX) {
                widened.uncompressedSize = Number(view.getBigUint64(q, true));
                q += 8;
            }
            if (widened.compressedSize === U32_MAX) {
                widened.compressedSize = Number(view.getBigUint64(q, true));
                q += 8;
            }
            if (widened.localOffset === U32_MAX) {
                widened.localOffset = Number(view.getBigUint64(q, true));
                q += 8;
            }
            return widened;
        }
        p += 4 + size;
    }
    throw new Error("Corrupt ZIP64 archive (entry has no Zip64 extra field).");
}

async function inflateRaw(
    bytes: Uint8Array<ArrayBuffer>,
): Promise<Uint8Array<ArrayBuffer>> {
    const stream = new Blob([bytes])
        .stream()
        .pipeThrough(new DecompressionStream("deflate-raw"));
    const output = await new Response(stream).arrayBuffer();
    return new Uint8Array(output);
}
