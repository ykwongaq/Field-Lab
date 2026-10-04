import { useCallback, useMemo, useState } from "react";
import type { Clip } from "../lib/clip";
import type { FrameSource } from "../lib/frames";
import type { LabelStore } from "../lib/labelStore";
import type { ZipArchive } from "../lib/zip";
import { downloadBlob } from "../lib/zipWriter";
import {
    archiveFrameEntries,
    exportAnnotation,
    exportOriginalFrames,
    exportProjectArchive,
    exportSampledFrames,
    exportSourceVideo,
    sourceVideoEntry,
    type ExportedFile,
    type ExportProgress,
} from "../lib/exporters";
import type { ExportOption } from "../components/ExportMenu";

export interface UseExportParams {
    clip: Clip;
    frames: FrameSource;
    zip: ZipArchive;
    store: LabelStore;
}

export interface ExportController {
    exportOpen: boolean;
    exportBusyId: string | null;
    exportProgress: string | null;
    exportError: string | null;
    exportOptions: ExportOption[];
    /** Clear any previous error and open the chooser. */
    openExport: () => void;
    closeExport: () => void;
}

/**
 * The workspace's export chooser.
 *
 * Owns the dialog's own state, the "write one choice" runner and the options
 * the clip actually supports. An archive packed from a video has no frame
 * folder, one packed from frames has no video, and a video-only archive records
 * no frame names at all — hence the per-option availability checks.
 *
 * A finished export only closes the chooser: there is no success toast.
 */
export function useExport({
    clip,
    frames,
    zip,
    store,
}: UseExportParams): ExportController {
    const [exportOpen, setExportOpen] = useState(false);
    const [exportBusyId, setExportBusyId] = useState<string | null>(null);
    const [exportProgress, setExportProgress] = useState<string | null>(null);
    const [exportError, setExportError] = useState<string | null>(null);

    /**
     * Run one export choice.
     *
     * The menu stays open while a multi-file export is written (progress lands
     * in `busyText`) so a large clip shows something rather than looking stuck.
     */
    const runExport = useCallback(
        async (
            id: string,
            unit: string,
            task: (
                report: ExportProgress,
            ) => ExportedFile | Promise<ExportedFile>,
        ) => {
            setExportError(null);
            setExportBusyId(id);
            setExportProgress(null);
            try {
                const file = await task((done, total) =>
                    setExportProgress(`${done} / ${total} ${unit}`.trim()),
                );
                downloadBlob(file.fileName, file.blob);
                setExportOpen(false);
            } catch (cause) {
                setExportError(
                    cause instanceof Error ? cause.message : String(cause),
                );
            } finally {
                setExportBusyId(null);
                setExportProgress(null);
            }
        },
        [],
    );

    // What this project can actually hand back.
    const sourceVideo = useMemo(() => sourceVideoEntry(clip, zip), [clip, zip]);
    const packedFrames = useMemo(() => archiveFrameEntries(zip), [zip]);
    const semantic = clip.mode === "semantic";

    const exportOptions = useMemo<ExportOption[]>(
        () => [
            {
                id: "video",
                title: "Original video",
                detail: "The source video this project was packed from, copied out unrecompressed.",
                ...(sourceVideo
                    ? {}
                    : {
                          disabledReason:
                              "This project was packed from frames, so it carries no source video.",
                      }),
                run: () =>
                    runExport("video", "", () => exportSourceVideo(clip, zip)),
            },
            {
                id: "original-frames",
                title: "Original frames",
                detail: `The ${packedFrames.length} frame${
                    packedFrames.length === 1 ? "" : "s"
                } the archive carried, as a ZIP.`,
                ...(packedFrames.length
                    ? {}
                    : {
                          disabledReason:
                              "This project was packed from a video, so it carries no frame folder.",
                      }),
                run: () =>
                    runExport("original-frames", "frames", (report) =>
                        exportOriginalFrames(clip, zip, report),
                    ),
            },
            {
                id: "sampled-frames",
                title: "Sampled frames",
                detail: `The ${frames.count} frame${
                    frames.count === 1 ? "" : "s"
                } this review annotated, as a ZIP.`,
                ...(frames.count
                    ? {}
                    : { disabledReason: "This project has no frames." }),
                run: () =>
                    runExport("sampled-frames", "frames", (report) =>
                        exportSampledFrames(clip, frames, report),
                    ),
            },
            semantic
                ? {
                      id: "project",
                      title: "Updated project archive",
                      detail: "A .project holding the frames, the updated label maps and the annotation — reopen it to carry on from here.",
                      run: () =>
                          runExport("project", "label maps", (report) =>
                              exportProjectArchive(
                                  clip,
                                  zip,
                                  store.getRecord(),
                                  report,
                              ),
                          ),
                  }
                : {
                      id: "annotation",
                      title: "Annotation JSON",
                      detail: "The dataset as it stands: one entry per tracklet with its per-frame RLE masks.",
                      run: () =>
                          runExport("annotation", "frames", () =>
                              exportAnnotation(clip, store.getRecord()),
                          ),
                  },
        ],
        [
            clip,
            frames,
            packedFrames.length,
            runExport,
            semantic,
            sourceVideo,
            store,
            zip,
        ],
    );

    const openExport = useCallback(() => {
        setExportError(null);
        setExportOpen(true);
    }, []);
    const closeExport = useCallback(() => setExportOpen(false), []);

    return {
        exportOpen,
        exportBusyId,
        exportProgress,
        exportError,
        exportOptions,
        openExport,
        closeExport,
    };
}
