import {
    useCallback,
    useEffect,
    useId,
    useMemo,
    useRef,
    useState,
} from "react";
import { filesFromDrop } from "../lib/droppedFiles";
import {
    LOCKED_MODES,
    MODE_VOCABULARY,
    PROJECT_MODES,
    type ProjectMode,
} from "../lib/project";
import {
    createProjectFile,
    naturalCompare,
    type CreateProjectProgress,
    type CreatedProject,
} from "../lib/projectWriter";
import { formatDuration, probeVideo, type VideoInfo } from "../lib/videoProbe";
import { downloadBlob } from "../lib/zipWriter";
import { LockIcon } from "./LockIcon";
import styles from "./CreateWizard.module.css";

/**
 * Three steps: pick a source, describe it, get the `.project` file.
 *
 * Step 1 keeps the two ways in side by side (a video, or a folder of frames).
 * Step 2 is the review of what was chosen — the video plays in the middle with
 * the settings underneath. Step 3 hands over the archive and offers to start
 * again. Nothing is uploaded at any point; the packing happens in the browser.
 */

const VIDEO_ACCEPT = "video/*,.mp4,.mov,.m4v,.avi,.mkv,.webm,.mpg,.mpeg";
const IMAGE_ACCEPT = "image/*,.jpg,.jpeg,.png,.webp,.bmp,.tif,.tiff";
const DEFAULT_TARGET_FPS = 6;
const VIDEO_EXTENSIONS = [
    ".mp4",
    ".mov",
    ".m4v",
    ".avi",
    ".mkv",
    ".webm",
    ".mpg",
    ".mpeg",
];

type Step = 1 | 2 | 3 | 4;

/** Which frame rate the project is built for: the measured one, or a typed one. */
type FpsChoice = "original" | "custom";

const STEPS: { step: Step; label: string }[] = [
    { step: 1, label: "Source" },
    { step: 2, label: "Details" },
    { step: 3, label: "Annotation" },
    { step: 4, label: "Save" },
];

export interface CreateWizardProps {
    /** Hand a finished archive to the app so it can be reviewed right away. */
    onOpen: (file: File) => void;
}

function stem(fileName: string): string {
    return basename(fileName).replace(/\.[^.]+$/, "");
}

function basename(name: string): string {
    return name.replace(/\\/g, "/").split("/").pop() ?? name;
}

function isVideoFile(file: File): boolean {
    const name = file.name.toLowerCase();
    return VIDEO_EXTENSIONS.some((extension) => name.endsWith(extension));
}

function formatBytes(bytes: number): string {
    if (bytes < 1024 ** 2) return `${Math.round(bytes / 1024)} KB`;
    if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
    return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

/** Object URL for a file, revoked when the file changes or the view unmounts. */
function useObjectUrl(file: File | null | undefined): string | null {
    const [url, setUrl] = useState<string | null>(null);
    useEffect(() => {
        if (!file) {
            setUrl(null);
            return;
        }
        const next = URL.createObjectURL(file);
        setUrl(next);
        return () => URL.revokeObjectURL(next);
    }, [file]);
    return url;
}

export function CreateWizard({ onOpen }: CreateWizardProps) {
    const videoInput = useRef<HTMLInputElement>(null);
    const framesInput = useRef<HTMLInputElement>(null);
    const abortRef = useRef<AbortController | null>(null);
    const nameId = useId();
    const customFpsId = useId();
    const annotationId = useId();

    const [step, setStep] = useState<Step>(1);
    const [video, setVideo] = useState<File | null>(null);
    const [frames, setFrames] = useState<File[] | null>(null);
    const [dragOver, setDragOver] = useState(false);
    const [notice, setNotice] = useState<string | null>(null);

    const [mode, setMode] = useState<ProjectMode | null>(null);
    const [name, setName] = useState("");
    const [fpsChoice, setFpsChoice] = useState<FpsChoice>("original");
    const [customFps, setCustomFps] = useState(DEFAULT_TARGET_FPS);
    const [annotation, setAnnotation] = useState<File | null>(null);

    const [info, setInfo] = useState<VideoInfo | null>(null);
    const [probing, setProbing] = useState(false);
    const [previewFailed, setPreviewFailed] = useState(false);

    const [created, setCreated] = useState<CreatedProject | null>(null);
    const [progress, setProgress] = useState<CreateProjectProgress | null>(
        null,
    );
    const [error, setError] = useState<string | null>(null);

    // Frames are stored in natural order, so the preview shows the same first
    // frame the reviewer will see.
    const orderedFrames = useMemo(
        () =>
            frames
                ? [...frames].sort((a, b) => naturalCompare(a.name, b.name))
                : null,
        [frames],
    );
    const previewUrl = useObjectUrl(video);
    const framePreviewUrl = useObjectUrl(orderedFrames?.[0] ?? null);

    // Read the video's duration, size and frame rate while step 2 is on screen.
    useEffect(() => {
        if (step !== 2 || !video) return;
        let cancelled = false;
        setProbing(true);
        probeVideo(video)
            .then((measured) => {
                if (cancelled) return;
                setInfo(measured);
                // Offer the measured rate; fall back to typing one.
                setFpsChoice(measured.fps ? "original" : "custom");
            })
            .catch(() => undefined)
            .finally(() => {
                if (!cancelled) setProbing(false);
            });
        return () => {
            cancelled = true;
        };
    }, [step, video]);

    const hasSource = video !== null || (frames?.length ?? 0) > 0;

    const setVideoSource = useCallback((file: File) => {
        setVideo(file);
        setFrames(null);
        setInfo(null);
        setPreviewFailed(false);
        // Optimistic: the probe usually measures it a moment later.
        setFpsChoice("original");
        setName((current) => (current.trim() ? current : stem(file.name)));
    }, []);

    const setFramesSource = useCallback((files: File[]) => {
        const images = files.filter((file) => !isVideoFile(file));
        if (images.length === 0) {
            setNotice("That folder holds no images (jpg, png, webp…).");
            return;
        }
        setFrames(images);
        setVideo(null);
        setInfo(null);
        // A folder of frames carries no rate of its own to measure.
        setFpsChoice("custom");
        setNotice(null);
        setName((current) => (current.trim() ? current : "clip"));
    }, []);

    /** Route a drop by its content, whichever card it landed on. */
    const handleDrop = useCallback(
        async (transfer: DataTransfer) => {
            setDragOver(false);
            const files = await filesFromDrop(transfer);
            if (files.length === 0) {
                setNotice("Nothing readable was dropped.");
                return;
            }
            if (files.length === 1 && isVideoFile(files[0])) {
                setVideoSource(files[0]);
                return;
            }
            setFramesSource(files);
        },
        [setFramesSource, setVideoSource],
    );

    const startOver = useCallback(() => {
        abortRef.current?.abort();
        setStep(1);
        setVideo(null);
        setFrames(null);
        setMode(null);
        setName("");
        setFpsChoice("original");
        setCustomFps(DEFAULT_TARGET_FPS);
        setAnnotation(null);
        setInfo(null);
        setCreated(null);
        setProgress(null);
        setError(null);
        setNotice(null);
    }, []);

    /** The rate the browser measured in the source, when it managed to. */
    const originalFps = info?.fps ?? null;
    const frameRate =
        fpsChoice === "original" && originalFps !== null
            ? originalFps
            : customFps;
    const trimmedName = name.trim();
    /** Step 2 is complete once a mode is picked and the project is named. */
    const detailsComplete = mode !== null && trimmedName !== "";
    const canCreate = detailsComplete && progress === null;

    const handleCreate = useCallback(async () => {
        if (!mode || !hasSource || !canCreate) return;
        setError(null);
        setProgress(null);
        const controller = new AbortController();
        abortRef.current = controller;
        try {
            const result = await createProjectFile({
                name: trimmedName,
                mode,
                video,
                frames,
                annotation,
                originalFps,
                targetFps: frameRate,
                width: info?.width ?? null,
                height: info?.height ?? null,
                signal: controller.signal,
                onProgress: setProgress,
            });
            downloadBlob(result.fileName, result.blob);
            setCreated(result);
            setStep(4);
        } catch (cause) {
            if (cause instanceof DOMException && cause.name === "AbortError") {
                return;
            }
            setError(cause instanceof Error ? cause.message : String(cause));
        } finally {
            abortRef.current = null;
            setProgress(null);
        }
    }, [
        annotation,
        canCreate,
        frameRate,
        frames,
        hasSource,
        info,
        mode,
        originalFps,
        trimmedName,
        video,
    ]);

    const stepList = (
        <ol className={styles.steps}>
            {STEPS.map((entry) => (
                <li
                    key={entry.step}
                    className={`${styles.step} ${
                        entry.step === step ? styles.stepActive : ""
                    } ${entry.step < step ? styles.stepDone : ""}`}
                >
                    <span className={styles.stepNumber}>{entry.step}</span>
                    {entry.label}
                </li>
            ))}
        </ol>
    );

    // ------------------------------------------------------------ step 1: source

    if (step === 1) {
        return (
            <section className={styles.panel}>
                {stepList}
                <h2 className={styles.heading}>Choose what to pack</h2>
                <p className={styles.lede}>
                    A video, or an existing folder of frames. Either is stored
                    as it is — nothing is decoded and nothing is uploaded.
                </p>

                <div className={styles.sourceGrid}>
                    <div
                        role="button"
                        tabIndex={0}
                        className={`${styles.videoDrop} ${
                            video ? styles.hasFile : ""
                        } ${dragOver ? styles.dragOver : ""}`}
                        onClick={() => videoInput.current?.click()}
                        onKeyDown={(event) => {
                            if (event.key === "Enter" || event.key === " ") {
                                event.preventDefault();
                                videoInput.current?.click();
                            }
                        }}
                        onDragOver={(event) => {
                            event.preventDefault();
                            setDragOver(true);
                        }}
                        onDragLeave={() => setDragOver(false)}
                        onDrop={(event) => {
                            event.preventDefault();
                            void handleDrop(event.dataTransfer);
                        }}
                    >
                        <span className={styles.dropTitle}>Source video</span>
                        {video ? (
                            <>
                                <span className={styles.fileName}>
                                    {basename(video.name)}
                                </span>
                                <span className={styles.fileMeta}>
                                    {formatBytes(video.size)}
                                </span>
                            </>
                        ) : (
                            <span className={styles.fileMeta}>
                                drop a file, or click to browse
                                <br />
                                mp4 · mov · mkv · webm · avi
                            </span>
                        )}
                    </div>

                    <div
                        role="button"
                        tabIndex={0}
                        className={`${styles.videoDrop} ${
                            frames ? styles.hasFile : ""
                        } ${dragOver ? styles.dragOver : ""}`}
                        onClick={() => framesInput.current?.click()}
                        onKeyDown={(event) => {
                            if (event.key === "Enter" || event.key === " ") {
                                event.preventDefault();
                                framesInput.current?.click();
                            }
                        }}
                        onDragOver={(event) => {
                            event.preventDefault();
                            setDragOver(true);
                        }}
                        onDragLeave={() => setDragOver(false)}
                        onDrop={(event) => {
                            event.preventDefault();
                            void handleDrop(event.dataTransfer);
                        }}
                    >
                        <span className={styles.dropTitle}>
                            Folder of frames
                        </span>
                        {frames ? (
                            <>
                                <span className={styles.fileName}>
                                    {frames.length} frames
                                </span>
                                <span className={styles.fileMeta}>
                                    {formatBytes(
                                        frames.reduce(
                                            (total, file) => total + file.size,
                                            0,
                                        ),
                                    )}{" "}
                                    · from {basename(frames[0].name)}
                                </span>
                            </>
                        ) : (
                            <span className={styles.fileMeta}>
                                drop the folder, or click to browse
                                <br />
                                jpg · png · webp · bmp · tif
                            </span>
                        )}
                    </div>
                </div>

                <input
                    ref={videoInput}
                    type="file"
                    accept={VIDEO_ACCEPT}
                    className={styles.hiddenInput}
                    onChange={(event) => {
                        const file = event.target.files?.[0];
                        if (file) setVideoSource(file);
                        event.target.value = "";
                    }}
                />
                <input
                    ref={framesInput}
                    type="file"
                    multiple
                    accept={IMAGE_ACCEPT}
                    className={styles.hiddenInput}
                    {...({
                        webkitdirectory: "",
                        directory: "",
                    } as Record<string, string>)}
                    onChange={(event) => {
                        const files = event.target.files
                            ? Array.from(event.target.files)
                            : [];
                        if (files.length > 0) setFramesSource(files);
                        event.target.value = "";
                    }}
                />

                {notice && <p className={styles.fileMeta}>{notice}</p>}

                <div className={styles.actions}>
                    <button
                        type="button"
                        className="btn btnPrimary"
                        disabled={!hasSource}
                        onClick={() => setStep(2)}
                    >
                        Continue
                    </button>
                    {hasSource && (
                        <button
                            type="button"
                            className="btn"
                            onClick={() => {
                                setVideo(null);
                                setFrames(null);
                                setNotice(null);
                            }}
                        >
                            Clear
                        </button>
                    )}
                    <span className={styles.summary}>
                        {hasSource
                            ? "Next: name it, set the frame rate and add annotations."
                            : "Waiting for a video or a folder of frames."}
                    </span>
                </div>
            </section>
        );
    }

    // ----------------------------------------------------------- step 2: details

    if (step === 2) {
        return (
            <section className={styles.panel}>
                {stepList}
                <h2 className={styles.heading}>Describe the project</h2>

                <div className={styles.detailsGrid}>
                    <div className={styles.preview}>
                        {video && previewUrl ? (
                            <video
                                className={styles.previewMedia}
                                src={previewUrl}
                                controls
                                preload="metadata"
                                onLoadedData={() => setPreviewFailed(false)}
                                onError={() => setPreviewFailed(true)}
                            />
                        ) : framePreviewUrl ? (
                            <img
                                className={styles.previewMedia}
                                src={framePreviewUrl}
                                alt="First frame of the selection"
                            />
                        ) : null}
                        <p className={styles.fileMeta}>
                            {video
                                ? `${basename(video.name)} · ${formatBytes(video.size)}`
                                : `${frames?.length ?? 0} frames${
                                      orderedFrames?.[0]
                                          ? ` · first: ${basename(orderedFrames[0].name)}`
                                          : ""
                                  }`}
                            {info
                                ? ` · ${formatDuration(info.durationSeconds)} · ${info.width}×${info.height}`
                                : ""}
                            {probing ? " · reading the video…" : ""}
                        </p>
                        {previewFailed && (
                            <p className={styles.previewNote}>
                                This browser cannot play this file back —
                                containers like mkv and avi usually need
                                converting first. It is packed as it is, so the
                                project is unaffected; only the preview and the
                                fps reading are missing.
                            </p>
                        )}
                    </div>

                    <div className={styles.settings}>
                        <fieldset className={styles.modes}>
                            <legend className={styles.legend}>
                                Segmentation mode
                                <span className={styles.lockNote}>
                                    <LockIcon /> fixed at creation, cannot be
                                    changed later
                                </span>
                            </legend>
                            <div className={styles.modeGrid}>
                                {PROJECT_MODES.map((option) => {
                                    const vocabulary = MODE_VOCABULARY[option];
                                    const locked =
                                        LOCKED_MODES.includes(option);
                                    const active = mode === option;
                                    return (
                                        <label
                                            key={option}
                                            className={`${styles.modeCard} ${
                                                active ? styles.modeActive : ""
                                            } ${
                                                locked ? styles.modeLocked : ""
                                            }`}
                                            title={
                                                locked
                                                    ? "Not available yet — semantic projects are temporarily locked."
                                                    : undefined
                                            }
                                        >
                                            <input
                                                type="radio"
                                                name="mode"
                                                value={option}
                                                checked={active}
                                                onChange={() => setMode(option)}
                                                className={styles.modeRadio}
                                                disabled={locked}
                                                required
                                            />
                                            <span className={styles.modeTitle}>
                                                {vocabulary.title}
                                                {locked ? " (unavailable)" : ""}
                                            </span>
                                            <span
                                                className={
                                                    styles.modeDescription
                                                }
                                            >
                                                {vocabulary.description}
                                            </span>
                                        </label>
                                    );
                                })}
                            </div>
                        </fieldset>

                        <label className={styles.option} htmlFor={nameId}>
                            <span className={styles.optionLabel}>
                                Project name
                            </span>
                            <input
                                id={nameId}
                                className={styles.textInput}
                                value={name}
                                placeholder="clip_007"
                                onChange={(event) =>
                                    setName(event.target.value)
                                }
                            />
                        </label>

                        <div className={styles.fpsBlock}>
                            <span className={styles.optionLabel}>
                                Frame rate
                            </span>
                            <div
                                className={styles.fpsChoices}
                                role="group"
                                aria-label="Frame rate"
                            >
                                <button
                                    type="button"
                                    className={`btn ${fpsChoice === "original" ? "btnPrimary" : ""}`}
                                    aria-pressed={fpsChoice === "original"}
                                    disabled={originalFps === null}
                                    title={
                                        originalFps === null
                                            ? "The browser could not measure the source rate"
                                            : undefined
                                    }
                                    onClick={() => setFpsChoice("original")}
                                >
                                    {probing
                                        ? "Measuring the source…"
                                        : originalFps !== null
                                          ? `Use the original · ${originalFps} fps`
                                          : "Original fps unavailable"}
                                </button>
                                <button
                                    type="button"
                                    className={`btn ${fpsChoice === "custom" ? "btnPrimary" : ""}`}
                                    aria-pressed={fpsChoice === "custom"}
                                    onClick={() => setFpsChoice("custom")}
                                >
                                    Enter a frame rate
                                </button>
                                {fpsChoice === "custom" && (
                                    <span className={styles.stepGroup}>
                                        <input
                                            id={customFpsId}
                                            type="number"
                                            min={1}
                                            max={240}
                                            step={1}
                                            className={styles.numberInput}
                                            value={customFps}
                                            onChange={(event) =>
                                                setCustomFps(
                                                    Math.max(
                                                        1,
                                                        Math.min(
                                                            240,
                                                            Number(
                                                                event.target
                                                                    .value,
                                                            ) || 1,
                                                        ),
                                                    ),
                                                )
                                            }
                                        />
                                        <span className={styles.stepSuffix}>
                                            fps
                                        </span>
                                    </span>
                                )}
                            </div>
                            <p className={styles.fileMeta}>
                                {originalFps !== null
                                    ? `The browser measured ${originalFps} fps in the source. `
                                    : "This file's rate could not be measured. "}
                                The project is built for {""}
                                <strong>{frameRate} fps</strong>.
                            </p>
                        </div>

                        <div className={styles.actions}>
                            <button
                                type="button"
                                className="btn btnPrimary"
                                disabled={!detailsComplete}
                                onClick={() => setStep(3)}
                            >
                                Continue
                            </button>
                            <button
                                type="button"
                                className="btn"
                                onClick={() => setStep(1)}
                            >
                                Back
                            </button>
                            {!detailsComplete && (
                                <span className={styles.summary}>
                                    {!mode
                                        ? "Choose a segmentation mode."
                                        : "Give the project a name."}
                                </span>
                            )}
                        </div>
                    </div>
                </div>
            </section>
        );
    }

    // ------------------------------------------------------ step 3: annotation

    if (step === 3) {
        return (
            <section className={styles.panel}>
                {stepList}
                <h2 className={styles.heading}>Annotation</h2>
                <p className={styles.lede}>
                    Optional. Seed the project from an existing
                    VideoSegmentation JSON — its video record is completed to
                    match the frames — or skip it and start empty.
                </p>

                <label className={styles.option} htmlFor={annotationId}>
                    <span className={styles.optionLabel}>
                        Annotation JSON (optional)
                    </span>
                    <input
                        id={annotationId}
                        type="file"
                        accept="application/json,.json"
                        className={styles.textInput}
                        onChange={(event) => {
                            setAnnotation(event.target.files?.[0] ?? null);
                            event.target.value = "";
                        }}
                    />
                </label>
                <p className={styles.fileMeta}>
                    {annotation
                        ? `Annotations will be seeded from ${basename(annotation.name)}.`
                        : "No file chosen: the project starts with no tracklets."}
                </p>

                <ul className={styles.resultList}>
                    <li>
                        <span className={styles.resultKey}>Source</span>
                        <span>
                            {video
                                ? `${basename(video.name)} · ${formatBytes(video.size)}`
                                : `${frames?.length ?? 0} frames`}
                        </span>
                    </li>
                    <li>
                        <span className={styles.resultKey}>Mode</span>
                        <span>{mode ? MODE_VOCABULARY[mode].title : "–"}</span>
                    </li>
                    <li>
                        <span className={styles.resultKey}>Frame rate</span>
                        <span>{frameRate} fps</span>
                    </li>
                    <li>
                        <span className={styles.resultKey}>
                            Will be saved as
                        </span>
                        <span className={styles.fileName}>
                            {trimmedName}.project
                        </span>
                    </li>
                </ul>

                {progress && (
                    <div
                        className={styles.progressTrack}
                        role="progressbar"
                        aria-valuemin={0}
                        aria-valuemax={100}
                        aria-valuenow={progressPercent(progress) ?? undefined}
                    >
                        <div
                            className={styles.progressFill}
                            style={
                                progressPercent(progress) === null
                                    ? undefined
                                    : {
                                          width: `${progressPercent(progress)}%`,
                                      }
                            }
                        />
                    </div>
                )}
                {error && <p className={styles.error}>{error}</p>}

                <div className={styles.actions}>
                    <button
                        type="button"
                        className="btn btnPrimary"
                        disabled={!canCreate}
                        onClick={() => void handleCreate()}
                    >
                        {progress ? "Packing…" : "Create project"}
                    </button>
                    <button
                        type="button"
                        className="btn"
                        disabled={progress !== null}
                        onClick={() => setStep(2)}
                    >
                        Back
                    </button>
                    {progress && (
                        <>
                            <button
                                type="button"
                                className="btn"
                                onClick={() => abortRef.current?.abort()}
                            >
                                Cancel
                            </button>
                            <span className={styles.summary}>
                                {describeProgress(progress)}
                            </span>
                        </>
                    )}
                </div>
            </section>
        );
    }

    // ------------------------------------------------------------- step 4: done

    return (
        <section className={styles.panel}>
            {stepList}
            <h2 className={styles.heading}>Project saved</h2>
            <p className={styles.lede}>
                <code>{created?.fileName}</code> has been downloaded. Put it
                somewhere safe — it is the whole project: frames, annotations
                and metadata in one file.
            </p>

            {created && (
                <ul className={styles.resultList}>
                    <li>
                        <span className={styles.resultKey}>Name</span>
                        <span className={styles.fileName}>{created.name}</span>
                    </li>
                    <li>
                        <span className={styles.resultKey}>Mode</span>
                        <span>
                            {MODE_VOCABULARY[created.mode].title} —{" "}
                            {MODE_VOCABULARY[created.mode].description}
                        </span>
                    </li>
                    <li>
                        <span className={styles.resultKey}>Contents</span>
                        <span>
                            {created.source === "video"
                                ? "the source video"
                                : `${created.frameCount} frames`}
                            {created.source === "video" &&
                            created.frameCount > 0
                                ? ` and ${created.frameCount} frames`
                                : ""}
                        </span>
                    </li>
                    <li>
                        <span className={styles.resultKey}>Frame rate</span>
                        <span>{created.fps} fps</span>
                    </li>
                    <li>
                        <span className={styles.resultKey}>Archive</span>
                        <span>{formatBytes(created.blob.size)}</span>
                    </li>
                </ul>
            )}

            <div className={styles.actions}>
                <button
                    type="button"
                    className="btn btnPrimary"
                    onClick={startOver}
                >
                    Create another project
                </button>
                {created && (
                    <button
                        type="button"
                        className="btn"
                        onClick={() =>
                            onOpen(
                                new File([created.blob], created.fileName, {
                                    type: "application/zip",
                                }),
                            )
                        }
                    >
                        Open it in the reviewer
                    </button>
                )}
                <span className={styles.summary}>
                    Opening it right away skips the download step; the file is
                    saved either way.
                </span>
            </div>
        </section>
    );
}

function progressPercent(progress: CreateProjectProgress): number | null {
    if (progress.stage === "video") {
        return progress.total
            ? Math.round((progress.loaded / progress.total) * 100)
            : null;
    }
    return progress.total
        ? Math.round((progress.done / progress.total) * 100)
        : null;
}

function describeProgress(progress: CreateProjectProgress): string {
    return progress.stage === "video"
        ? `Reading the video · ${Math.round(progress.loaded / 1024 ** 2)} MB`
        : `Adding frames · ${progress.done}/${progress.total}`;
}
