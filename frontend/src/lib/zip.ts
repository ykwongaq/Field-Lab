

const EOCD_SIGNATURE = 0x06054b50;
const CD_SIGNATURE = 0x02014b50;
const LFH_SIGNATURE = 0x04034b50;

interface InternalEntry {
	name: string;
	method: number;
	crc32: number;
	compressedSize: number;
	uncompressedSize: number;
	dataOffset: number;
}


export interface ZipRawEntry {
	name: string;
	method: number;
	crc32: number;
	compressedSize: number;
	uncompressedSize: number;
	data: Uint8Array;
}

export class ZipArchive {
	private readonly buffer: ArrayBuffer;
	private readonly entries: Map<string, InternalEntry>;

	private constructor(
		buffer: ArrayBuffer,
		entries: Map<string, InternalEntry>,
	) {
		this.buffer = buffer;
		this.entries = entries;
	}

	static async fromFile(file: File): Promise<ZipArchive> {
		const buffer = await file.arrayBuffer();
		const view = new DataView(buffer);
		const eocdOffset = findEocdOffset(view);
		const entryCount = view.getUint16(eocdOffset + 10, true);
		const cdOffset = view.getUint32(eocdOffset + 16, true);

		const decoder = new TextDecoder("utf-8");
		const entries = new Map<string, InternalEntry>();
		let p = cdOffset;

		for (let i = 0; i < entryCount; i++) {
			if (view.getUint32(p, true) !== CD_SIGNATURE) break;
			const method = view.getUint16(p + 10, true);
			const crc32 = view.getUint32(p + 16, true);
			const compressedSize = view.getUint32(p + 20, true);
			const uncompressedSize = view.getUint32(p + 24, true);
			const nameLength = view.getUint16(p + 28, true);
			const extraLength = view.getUint16(p + 30, true);
			const commentLength = view.getUint16(p + 32, true);
			const localOffset = view.getUint32(p + 42, true);
			const name = decoder.decode(new Uint8Array(buffer, p + 46, nameLength));
			const dataOffset = resolveDataOffset(view, localOffset);
			entries.set(name, {
				name,
				method,
				crc32,
				compressedSize,
				uncompressedSize,
				dataOffset,
			});
			p += 46 + nameLength + extraLength + commentLength;
		}

		return new ZipArchive(buffer, entries);
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
		const bytes = new Uint8Array(
			this.buffer,
			entry.dataOffset,
			entry.compressedSize,
		);

		if (entry.method === 0) return new Blob([bytes.slice()]);
		if (entry.method === 8) return new Blob([await inflateRaw(bytes)]);
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

	
	rawEntry(name: string): ZipRawEntry {
		const entry = this.entries.get(name);
		if (!entry) throw new Error(`Missing zip entry: ${name}`);
		return {
			name: entry.name,
			method: entry.method,
			crc32: entry.crc32,
			compressedSize: entry.compressedSize,
			uncompressedSize: entry.uncompressedSize,
			data: new Uint8Array(this.buffer, entry.dataOffset, entry.compressedSize),
		};
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

function resolveDataOffset(view: DataView, localOffset: number): number {
	if (view.getUint32(localOffset, true) !== LFH_SIGNATURE) {
		throw new Error("Corrupt ZIP local file header.");
	}
	const nameLength = view.getUint16(localOffset + 26, true);
	const extraLength = view.getUint16(localOffset + 28, true);
	return localOffset + 30 + nameLength + extraLength;
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
