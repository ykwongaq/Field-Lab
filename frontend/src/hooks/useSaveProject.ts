import { useCallback, useState } from "react";
import type { Clip } from "../lib/clip";
import type { LabelStore } from "../lib/labelStore";
import type { ZipArchive } from "../lib/zip";
import { downloadBlob } from "../lib/zipWriter";
import { exportProjectArchive } from "../lib/exporters";

export interface UseSaveProjectParams {
    clip: Clip;
    zip: ZipArchive;
    store: LabelStore;
}

/** What a save attempt produced, so the caller can report it. */
export type SaveProjectResult =
    | { ok: true; fileName: string }
    | { ok: false; error: string };

export interface SaveProjectController {
    /** `true` while the archive is being packed and handed to the browser. */
    saveBusy: boolean;
    /** Progress read-out while label maps are written, e.g. "3 / 10". */
    saveProgress: string | null;
    /** Pack the current state into a `.project` archive and download it. */
    saveProject: () => Promise<SaveProjectResult>;
}

/**
 * The workspace's save action.
 *
 * Saving packs the current state back into a `.project` archive — the same
 * layout the project was opened from — and hands it to the browser as a
 * download, so the review can be reopened later. Nothing is uploaded: every byte
 * already lives in the browser (the archive, the frames and the `Clip`).
 *
 * The result is returned rather than raised so the caller owns how it is
 * reported; the hook only tracks the busy and progress read-outs.
 */
export function useSaveProject({
    clip,
    zip,
    store,
}: UseSaveProjectParams): SaveProjectController {
    const [saveBusy, setSaveBusy] = useState(false);
    const [saveProgress, setSaveProgress] = useState<string | null>(null);

    const saveProject = useCallback(async (): Promise<SaveProjectResult> => {
        setSaveBusy(true);
        setSaveProgress(null);
        try {
            const file = await exportProjectArchive(
                clip,
                zip,
                store.getRecord(),
                (done, total) =>
                    setSaveProgress(`${done} / ${total} label maps`),
            );
            downloadBlob(file.fileName, file.blob);
            return { ok: true, fileName: file.fileName };
        } catch (cause) {
            return {
                ok: false,
                error: cause instanceof Error ? cause.message : String(cause),
            };
        } finally {
            setSaveBusy(false);
            setSaveProgress(null);
        }
    }, [clip, store, zip]);

    return { saveBusy, saveProgress, saveProject };
}
