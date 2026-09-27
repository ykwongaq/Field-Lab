import type { ZipRawEntry } from "./zip";


const LFH_SIGNATURE = 0x04034b50;
const CD_SIGNATURE = 0x02014b50;
const EOCD_SIGNATURE = 0x06054b50;

const CRC_TABLE = (() => {
	const table = new Uint32Array(256);
	for (let n = 0; n < 256; n++) {
		let c = n;
		for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
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

interface PendingEntry {
	nameBytes: Uint8Array;
	method: number;
	crc32: number;
	compressedSize: number;
	uncompressedSize: number;
	data: Uint8Array | Blob;
	localOffset: number;
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

	async addBlob(name: string, blob: Blob): Promise<void> {
		this.addStored(name, new Uint8Array(await blob.arrayBuffer()));
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
		fields: Omit<PendingEntry, "nameBytes" | "localOffset">,
	): void {
		if (this.names.has(name)) throw new Error(`Duplicate zip entry: ${name}`);
		this.names.add(name);
		const nameBytes = this.encoder.encode(name);
		this.entries.push({ ...fields, nameBytes, localOffset: this.offset });
		this.offset += 30 + nameBytes.length + fields.compressedSize;
	}

	/** Assemble the archive. */
	toBlob(): Blob {
		const parts: BlobPart[] = [];
		for (const entry of this.entries) {
			const header = new DataView(new ArrayBuffer(30));
			header.setUint32(0, LFH_SIGNATURE, true);
			header.setUint16(4, 20, true); // version needed: 2.0
			header.setUint16(6, 0x0800, true); // flags: UTF-8 names
			header.setUint16(8, entry.method, true);
			header.setUint16(10, this.stamp.time, true);
			header.setUint16(12, this.stamp.date, true);
			header.setUint32(14, entry.crc32, true);
			header.setUint32(18, entry.compressedSize, true);
			header.setUint32(22, entry.uncompressedSize, true);
			header.setUint16(26, entry.nameBytes.length, true);
			header.setUint16(28, 0, true);
			parts.push(header.buffer, entry.nameBytes as BlobPart, entry.data as BlobPart);
		}

		const cdStart = this.offset;
		let cdSize = 0;
		for (const entry of this.entries) {
			const record = new DataView(new ArrayBuffer(46));
			record.setUint32(0, CD_SIGNATURE, true);
			record.setUint16(4, 0x0314, true); // made by: UNIX, 2.0
			record.setUint16(6, 20, true);
			record.setUint16(8, 0x0800, true);
			record.setUint16(10, entry.method, true);
			record.setUint16(12, this.stamp.time, true);
			record.setUint16(14, this.stamp.date, true);
			record.setUint32(16, entry.crc32, true);
			record.setUint32(20, entry.compressedSize, true);
			record.setUint32(24, entry.uncompressedSize, true);
			record.setUint16(28, entry.nameBytes.length, true);
			record.setUint16(30, 0, true); // extra
			record.setUint16(32, 0, true); // comment
			record.setUint16(34, 0, true); // disk
			record.setUint16(36, 0, true); // internal attrs
			record.setUint32(38, 0x81a40000, true); // external attrs: -rw-r--r--
			record.setUint32(42, entry.localOffset, true);
			parts.push(record.buffer, entry.nameBytes as BlobPart);
			cdSize += 46 + entry.nameBytes.length;
		}

		const eocd = new DataView(new ArrayBuffer(22));
		eocd.setUint32(0, EOCD_SIGNATURE, true);
		eocd.setUint16(4, 0, true);
		eocd.setUint16(6, 0, true);
		eocd.setUint16(8, this.entries.length, true);
		eocd.setUint16(10, this.entries.length, true);
		eocd.setUint32(12, cdSize, true);
		eocd.setUint32(16, cdStart, true);
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
