import { useCallback, useRef, useState } from "react";
import { Clip } from "./lib/clip";
import { ZipArchive } from "./lib/zip";
import { MODE_VOCABULARY } from "./lib/project";
import {
	createProject,
	saveBlob,
	type CreateProjectProgress,
} from "./lib/projectApi";
import { UploadScreen } from "./components/UploadScreen";
import type { CreateProjectRequest } from "./components/CreateProjectPanel";
import { Workspace, type WorkspaceNotice } from "./components/Workspace";
import styles from "./App.module.css";

type Phase = "upload" | "loading" | "creating" | "ready" | "error";

function describeProgress(progress: CreateProjectProgress | null): {
	label: string;
	percent: number | null;
} {
	if (!progress) return { label: "Preparing upload…", percent: 0 };
	switch (progress.stage) {
		case "uploading": {
			const percent = progress.total
				? Math.round((progress.loaded / progress.total) * 100)
				: null;
			return {
				label: `Uploading video${percent !== null ? ` · ${percent}%` : ""}`,
				percent,
			};
		}
		case "processing":
			return {
				label: "Extracting frames and bundling the project archive…",
				percent: null,
			};
		case "downloading": {
			const percent = progress.total
				? Math.round((progress.loaded / progress.total) * 100)
				: null;
			return {
				label: `Receiving project archive${
					percent !== null ? ` · ${percent}%` : ""
				}`,
				percent,
			};
		}
	}
}

function App() {
	const [phase, setPhase] = useState<Phase>("upload");
	const [clip, setClip] = useState<Clip | null>(null);
	const [zip, setZip] = useState<ZipArchive | null>(null);
	const [error, setError] = useState<string | null>(null);
	const [notice, setNotice] = useState<WorkspaceNotice | null>(null);
	const [progress, setProgress] = useState<CreateProjectProgress | null>(null);
	const abortRef = useRef<AbortController | null>(null);

	const openArchive = useCallback(async (file: File) => {
		const archive = await ZipArchive.fromFile(file);
		const parsed = await Clip.fromZip(archive);
		setZip(archive);
		setClip(parsed);
		setPhase("ready");
	}, []);

	const handleFile = useCallback(
		async (file: File) => {
			setPhase("loading");
			setError(null);
			setNotice(null);
			try {
				await openArchive(file);
			} catch (cause) {
				setError(cause instanceof Error ? cause.message : String(cause));
				setPhase("error");
			}
		},
		[openArchive],
	);

	const handleCreate = useCallback(
		async (request: CreateProjectRequest) => {
			setPhase("creating");
			setError(null);
			setNotice(null);
			setProgress(null);
			const controller = new AbortController();
			abortRef.current = controller;
			try {
				const created = await createProject({
					video: request.video,
					mode: request.mode,
					name: request.name,
					frameStep: request.frameStep,
					signal: controller.signal,
					onProgress: setProgress,
				});
				// Persist the archive for the reviewer's records, then open it.
				saveBlob(created.blob, created.fileName);
				setPhase("loading");
				await openArchive(
					new File([created.blob], created.fileName, {
						type: "application/zip",
					}),
				);
				setNotice({
					kind: "success",
					text: `Project archive saved as ${created.fileName}. Segmentation mode is ${MODE_VOCABULARY[created.mode].title} and cannot be changed.`,
				});
			} catch (cause) {
				if (cause instanceof DOMException && cause.name === "AbortError") {
					setPhase("upload");
					return;
				}
				setError(cause instanceof Error ? cause.message : String(cause));
				setPhase("error");
			} finally {
				abortRef.current = null;
				setProgress(null);
			}
		},
		[openArchive],
	);

	const handleCancelCreate = useCallback(() => {
		abortRef.current?.abort();
	}, []);

	const handleReset = useCallback(() => {
		setClip(null);
		setZip(null);
		setError(null);
		setNotice(null);
		setPhase("upload");
	}, []);

	const progressView = describeProgress(progress);

	return (
		<div className={styles.app}>
			{phase === "upload" && (
				<UploadScreen onFile={handleFile} onCreate={handleCreate} />
			)}

			{phase === "loading" && (
				<div className={styles.center}>
					<h1 className={styles.title}>Loading</h1>
					<p className={styles.subtitle}>Reading frames and annotations…</p>
				</div>
			)}

			{phase === "creating" && (
				<div className={styles.center}>
					<h1 className={styles.title}>Creating project</h1>
					<p className={styles.subtitle}>{progressView.label}</p>
					<div
						className={`${styles.progressTrack} ${
							progressView.percent === null ? styles.progressIndeterminate : ""
						}`}
						role="progressbar"
						aria-valuemin={0}
						aria-valuemax={100}
						aria-valuenow={progressView.percent ?? undefined}
					>
						<div
							className={styles.progressFill}
							style={
								progressView.percent === null
									? undefined
									: { width: `${progressView.percent}%` }
							}
						/>
					</div>
					<button
						type="button"
						className="btn"
						onClick={handleCancelCreate}
					>
						Cancel
					</button>
				</div>
			)}

			{phase === "error" && (
				<div className={styles.center}>
					<h1 className={styles.title}>Could not open project</h1>
					<p className={styles.error}>{error}</p>
					<button
						type="button"
						className="btn btnPrimary"
						onClick={handleReset}
					>
						Back to start
					</button>
				</div>
			)}

			{phase === "ready" && clip !== null && zip !== null && (
				<Workspace
					key={clip.name}
					clip={clip}
					zip={zip}
					notice={notice}
					onDismissNotice={() => setNotice(null)}
					onReset={handleReset}
				/>
			)}
		</div>
	);
}

export default App;
