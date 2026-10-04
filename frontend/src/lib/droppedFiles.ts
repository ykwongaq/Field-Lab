/**
 * Read the files a drag-and-drop actually carried.
 *
 * A dropped *folder* shows up in `dataTransfer.files` as a single entry with no
 * contents, so the only way to walk it is `DataTransferItem.webkitGetAsEntry()`
 * plus directory readers. This flattens the whole tree into a list of files.
 */

/** Read one directory reader to exhaustion (it returns batches, not all rows). */
function readDirectory(
    reader: FileSystemDirectoryReader,
): Promise<FileSystemEntry[]> {
    return new Promise((resolve, reject) => {
        const batches: FileSystemEntry[] = [];
        const next = () => {
            reader.readEntries((entries) => {
                if (entries.length === 0) {
                    resolve(batches);
                    return;
                }
                batches.push(...entries);
                next();
            }, reject);
        };
        next();
    });
}

function fileOf(entry: FileSystemFileEntry): Promise<File | null> {
    return new Promise((resolve) => {
        entry.file(
            (file) => resolve(file),
            () => resolve(null),
        );
    });
}

async function collect(entry: FileSystemEntry, into: File[]): Promise<void> {
    if (entry.isFile) {
        const file = await fileOf(entry as FileSystemFileEntry);
        if (file) into.push(file);
        return;
    }
    if (!entry.isDirectory) return;
    const children = await readDirectory(
        (entry as FileSystemDirectoryEntry).createReader(),
    );
    for (const child of children) {
        await collect(child, into);
    }
}

/** The files of a drop, including everything inside a dropped folder. */
export async function filesFromDrop(transfer: DataTransfer): Promise<File[]> {
    const entries = Array.from(transfer.items ?? [])
        .filter((item) => item.kind === "file")
        .map((item) => item.webkitGetAsEntry?.() ?? null);

    if (entries.some((entry) => entry !== null)) {
        const files: File[] = [];
        for (const entry of entries) {
            if (entry) await collect(entry, files);
        }
        if (files.length > 0) return files;
    }
    return Array.from(transfer.files ?? []);
}
