import type { ZipRawEntry } from "./zip";

const LFH_SIGNATURE = 0x04034b50;
const CD_SIGNATURE = 0x02014b50;
const EOCD_SIGNATURE = 0x06054b50;

const ZIP64_EXTRA_ID = 0x0001;
const ZIP64_EOCD_SIGNATURE = 0x06064b50;
const ZIP64_LOCATOR_SIGNATURE = 0x07064b50;
/** Saturation values: a 32/16-bit field holding these points at a Zip64 record. */
const U32_MAX = 0xffffffff;
const U16_MAX = 0xffff;
/** "Version needed to extract" once an entry needs Zip64 fields. */
const ZIP64_VERSION = 45;
const ZIP64_LOCAL_EXTRA_LENGTH = 4 + 16; // header + uncompressed/compressed sizes

const CRC_TABLE = (() => {
    const table = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
        let c = n;
        for (let k = 0; k < 8; k++)
            c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
        table[n] = c >>> 0;
    }
    return table;
})();

export function crc32(bytes: Uint8Array): number {
    let crc = 0xffffffff;
    for (let i = 0; i < bytes.length; i++) {
        crc = CRC_TABLE[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
    }
    return (crc ^ 0xffffffff) >>> 0;
}

/**
 * CRC32 of a blob, read in streamed chunks.
 *
 * The point is to avoid `await blob.arrayBuffer()` on a multi-gigabyte file:
 * the bytes are consumed chunk by chunk, so peak memory stays flat no matter
 * how large the source video is. `onProgress` is called per chunk so the UI can
 * show something during a long checksum.
 */
export async function crc32OfBlob(
    blob: Blob,
    onProgress?: (loaded: number, total: number) => void,
): Promise<number> {
    let crc = 0xffffffff;
    let loaded = 0;
    const reader = blob.stream().getReader();
    try {
        for (;;) {
            const { done, value } = await reader.read();
            if (done) break;
            const bytes = value as Uint8Array;
            for (let i = 0; i < bytes.length; i++) {
                crc = CRC_TABLE[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
            }
            loaded += bytes.length;
            onProgress?.(loaded, blob.size);
        }
    } finally {
        await reader.cancel().catch(() => undefined);
    }
    return (crc ^ 0xffffffff) >>> 0;
}

interface PendingEntry {
    nameBytes: Uint8Array;
    method: number;
    crc32: number;
    compressedSize: number;
    uncompressedSize: number;
    data: Uint8Array | Blob;
    localOffset: number;
    /** Any 32-bit field above `U32_MAX`; the entry then emits Zip64 records. */
    zip64: boolean;
}

/**
 * The Zip64 "extended information" extra field for one entry.
 *
 * Only the fields that are saturated (0xFFFFFFFF) in the fixed-size record may
 * appear, in the order uncompressed size, compressed size, local offset — so
 * Zip64 entries simply write all of them.
 */
function zip64Extra(entry: PendingEntry, withOffset: boolean): Uint8Array {
    if (!entry.zip64) return new Uint8Array(0);
    const body = withOffset ? 24 : 16;
    const extra = new Uint8Array(4 + body);
    const view = new DataView(extra.buffer);
    view.setUint16(0, ZIP64_EXTRA_ID, true);
    view.setUint16(2, body, true);
    view.setBigUint64(4, BigInt(entry.uncompressedSize), true);
    view.setBigUint64(12, BigInt(entry.compressedSize), true);
    if (withOffset) view.setBigUint64(20, BigInt(entry.localOffset), true);
    return extra;
}

function dosDateTime(date: Date): { time: number; date: number } {
    const time =
        (date.getHours() << 11) |
        (date.getMinutes() << 5) |
        Math.floor(date.getSeconds() / 2);
    const day =
        ((Math.max(1980, date.getFullYear()) - 1980) << 9) |
        ((date.getMonth() + 1) << 5) |
        date.getDate();
    return { time, date: day };
}

export class ZipWriter {
    private readonly entries: PendingEntry[] = [];
    private readonly names = new Set<string>();
    private readonly encoder = new TextEncoder();
    private readonly stamp = dosDateTime(new Date());
    private offset = 0;

    addStored(name: string, data: Uint8Array): void {
        this.push(name, {
            method: 0,
            crc32: crc32(data),
            compressedSize: data.length,
            uncompressedSize: data.length,
            data,
        });
    }

    addText(name: string, text: string): void {
        this.addStored(name, this.encoder.encode(text));
    }

    /**
     * Store a blob as-is, without ever holding it in memory.
     *
     * A `File` from an `<input type="file">` is already a disk-backed `Blob`, so
     * only the CRC32 has to be read; the blob itself becomes one part of the
     * archive. That is what makes a multi-gigabyte source video zippable here.
     */
    async addBlob(
        name: string,
        blob: Blob,
        onProgress?: (loaded: number, total: number) => void,
    ): Promise<void> {
        const checksum = await crc32OfBlob(blob, onProgress);
        this.push(name, {
            method: 0,
            crc32: checksum,
            compressedSize: blob.size,
            uncompressedSize: blob.size,
            data: blob,
        });
    }

    addRaw(entry: ZipRawEntry): void {
        this.push(entry.name, {
            method: entry.method,
            crc32: entry.crc32,
            compressedSize: entry.compressedSize,
            uncompressedSize: entry.uncompressedSize,
            data: entry.data,
        });
    }

    has(name: string): boolean {
        return this.names.has(name);
    }

    private push(
        name: string,
        fields: Omit<PendingEntry, "nameBytes" | "localOffset" | "zip64">,
    ): void {
        if (this.names.has(name))
            throw new Error(`Duplicate zip entry: ${name}`);
        this.names.add(name);
        const nameBytes = this.encoder.encode(name);
        const localOffset = this.offset;
        const zip64 =
            fields.compressedSize > U32_MAX ||
            fields.uncompressedSize > U32_MAX ||
            localOffset > U32_MAX;
        this.entries.push({ ...fields, nameBytes, localOffset, zip64 });
        this.offset +=
            30 +
            nameBytes.length +
            (zip64 ? ZIP64_LOCAL_EXTRA_LENGTH : 0) +
            fields.compressedSize;
    }

    /** Assemble the archive. */
    toBlob(): Blob {
        const parts: BlobPart[] = [];
        for (const entry of this.entries) {
            const extra = zip64Extra(entry, false);
            const header = new DataView(new ArrayBuffer(30 + extra.length));
            header.setUint32(0, LFH_SIGNATURE, true);
            header.setUint16(4, entry.zip64 ? ZIP64_VERSION : 20, true);
            header.setUint16(6, 0x0800, true); // flags: UTF-8 names
            header.setUint16(8, entry.method, true);
            header.setUint16(10, this.stamp.time, true);
            header.setUint16(12, this.stamp.date, true);
            header.setUint32(14, entry.crc32, true);
            header.setUint32(
                18,
                entry.zip64 ? U32_MAX : entry.compressedSize,
                true,
            );
            header.setUint32(
                22,
                entry.zip64 ? U32_MAX : entry.uncompressedSize,
                true,
            );
            header.setUint16(26, entry.nameBytes.length, true);
            header.setUint16(28, extra.length, true);
            new Uint8Array(header.buffer, 30).set(extra);
            parts.push(
                header.buffer,
                entry.nameBytes as BlobPart,
                entry.data as BlobPart,
            );
        }

        const cdStart = this.offset;
        let cdSize = 0;
        for (const entry of this.entries) {
            const extra = zip64Extra(entry, true);
            const record = new DataView(new ArrayBuffer(46 + extra.length));
            record.setUint32(0, CD_SIGNATURE, true);
            record.setUint16(4, entry.zip64 ? 0x032d : 0x0314, true); // made by
            record.setUint16(6, entry.zip64 ? ZIP64_VERSION : 20, true);
            record.setUint16(8, 0x0800, true);
            record.setUint16(10, entry.method, true);
            record.setUint16(12, this.stamp.time, true);
            record.setUint16(14, this.stamp.date, true);
            record.setUint32(16, entry.crc32, true);
            record.setUint32(
                20,
                entry.zip64 ? U32_MAX : entry.compressedSize,
                true,
            );
            record.setUint32(
                24,
                entry.zip64 ? U32_MAX : entry.uncompressedSize,
                true,
            );
            record.setUint16(28, entry.nameBytes.length, true);
            record.setUint16(30, extra.length, true); // extra
            record.setUint16(32, 0, true); // comment
            record.setUint16(34, 0, true); // disk
            record.setUint16(36, 0, true); // internal attrs
            record.setUint32(38, 0x81a40000, true); // external attrs: -rw-r--r--
            record.setUint32(
                42,
                entry.zip64 ? U32_MAX : entry.localOffset,
                true,
            );
            new Uint8Array(record.buffer, 46).set(extra);
            parts.push(record.buffer, entry.nameBytes as BlobPart);
            cdSize += 46 + entry.nameBytes.length + extra.length;
        }

        // Zip64 needs its own directory whenever the classic record cannot hold
        // the entry count, the central-directory size or its offset.
        const needZip64 =
            this.entries.length > U16_MAX ||
            cdSize > U32_MAX ||
            cdStart > U32_MAX;
        if (needZip64) {
            const zipped = new DataView(new ArrayBuffer(56));
            zipped.setUint32(0, ZIP64_EOCD_SIGNATURE, true);
            zipped.setBigUint64(4, BigInt(44), true); // size of the rest
            zipped.setUint16(12, ZIP64_VERSION, true); // made by
            zipped.setUint16(14, ZIP64_VERSION, true); // needed to extract
            zipped.setUint32(16, 0, true); // this disk
            zipped.setUint32(20, 0, true); // disk with the directory
            zipped.setBigUint64(24, BigInt(this.entries.length), true);
            zipped.setBigUint64(32, BigInt(this.entries.length), true);
            zipped.setBigUint64(40, BigInt(cdSize), true);
            zipped.setBigUint64(48, BigInt(cdStart), true);
            parts.push(zipped.buffer);

            const locator = new DataView(new ArrayBuffer(20));
            locator.setUint32(0, ZIP64_LOCATOR_SIGNATURE, true);
            locator.setUint32(4, 0, true); // disk with the Zip64 record
            locator.setBigUint64(8, BigInt(cdStart + cdSize), true);
            locator.setUint32(16, 1, true); // total disks
            parts.push(locator.buffer);
        }

        const eocd = new DataView(new ArrayBuffer(22));
        eocd.setUint32(0, EOCD_SIGNATURE, true);
        eocd.setUint16(4, 0, true);
        eocd.setUint16(6, 0, true);
        eocd.setUint16(8, Math.min(this.entries.length, U16_MAX), true);
        eocd.setUint16(10, Math.min(this.entries.length, U16_MAX), true);
        eocd.setUint32(12, Math.min(cdSize, U32_MAX), true);
        eocd.setUint32(16, Math.min(cdStart, U32_MAX), true);
        eocd.setUint16(20, 0, true);
        parts.push(eocd.buffer);

        return new Blob(parts, { type: "application/zip" });
    }
}

export function downloadBlob(filename: string, blob: Blob): void {
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}
