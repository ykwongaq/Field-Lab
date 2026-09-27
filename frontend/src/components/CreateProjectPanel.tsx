import { useId, useRef, useState } from "react";
import {
	MODE_VOCABULARY,
	PROJECT_MODES,
	type ProjectMode,
} from "../lib/project";
import styles from "./CreateProjectPanel.module.css";

export interface CreateProjectRequest {
	video: File;
	mode: ProjectMode;
	name: string;
	frameStep: number;
}

interface CreateProjectPanelProps {
	onCreate: (request: CreateProjectRequest) => void;
	disabled?: boolean;
}

const VIDEO_ACCEPT =
	"video/*,.mp4,.mov,.m4v,.avi,.mkv,.webm,.mpg,.mpeg";

function stem(fileName: string): string {
	return fileName.replace(/\.[^.]+$/, "");
}

function formatBytes(bytes: number): string {
	if (bytes < 1024 ** 2) return `${Math.round(bytes / 1024)} KB`;
	if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
	return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}


export function CreateProjectPanel({
	onCreate,
	disabled = false,
}: CreateProjectPanelProps) {
	const inputRef = useRef<HTMLInputElement>(null);
	const nameId = useId();
	const stepId = useId();

	const [video, setVideo] = useState<File | null>(null);
	const [name, setName] = useState("");
	const [nameTouched, setNameTouched] = useState(false);
	const [mode, setMode] = useState<ProjectMode | null>(null);
	const [frameStep, setFrameStep] = useState(1);
	const [dragOver, setDragOver] = useState(false);

	const pickVideo = (file: File | undefined) => {
		if (!file) return;
		setVideo(file);
		if (!nameTouched) setName(stem(file.name));
	};

	const canSubmit = video !== null && mode !== null && !disabled;

	return (
		<form
			className={styles.panel}
			onSubmit={(event) => {
				event.preventDefault();
				if (!video || !mode) return;
				onCreate({ video, mode, name: name.trim(), frameStep });
			}}
		>
			<h2 className={styles.heading}>Create a project from a video</h2>
			<p className={styles.lede}>
				Upload one video. The backend extracts the frames and bundles them
				with the video and an empty annotation JSON into a single{" "}
				<code>.zip</code>, then opens it here.
			</p>

			{/* 1. Video */}
			<div
				role="button"
				tabIndex={0}
				aria-disabled={disabled}
				className={`${styles.videoDrop} ${dragOver ? styles.dragOver : ""} ${
					video ? styles.hasFile : ""
				}`}
				onClick={() => !disabled && inputRef.current?.click()}
				onKeyDown={(event) => {
					if (disabled) return;
					if (event.key === "Enter" || event.key === " ") {
						event.preventDefault();
						inputRef.current?.click();
					}
				}}
				onDragOver={(event) => {
					event.preventDefault();
					if (!disabled) setDragOver(true);
				}}
				onDragLeave={() => setDragOver(false)}
				onDrop={(event) => {
					event.preventDefault();
					setDragOver(false);
					if (!disabled) pickVideo(event.dataTransfer.files?.[0]);
				}}
			>
				{video ? (
					<>
						<span className={styles.fileName}>{video.name}</span>
						<span className={styles.fileMeta}>
							{formatBytes(video.size)} · click to replace
						</span>
					</>
				) : (
					<>
						<span className={styles.dropTitle}>Drop a video here</span>
						<span className={styles.fileMeta}>
							or click to browse · mp4, mov, mkv, webm, avi
						</span>
					</>
				)}
				<input
					ref={inputRef}
					type="file"
					accept={VIDEO_ACCEPT}
					className={styles.hiddenInput}
					disabled={disabled}
					onChange={(event) => {
						pickVideo(event.target.files?.[0]);
						event.target.value = "";
					}}
				/>
			</div>

			{/* 2. Mode — the one decision that cannot be undone */}
			<fieldset className={styles.modes} disabled={disabled}>
				<legend className={styles.legend}>
					Segmentation mode
					<span className={styles.lockNote}>
						<LockIcon /> fixed at creation, cannot be changed later
					</span>
				</legend>
				<div className={styles.modeGrid}>
					{PROJECT_MODES.map((option) => {
						const vocab = MODE_VOCABULARY[option];
						const active = mode === option;
						return (
							<label
								key={option}
								className={`${styles.modeCard} ${active ? styles.modeActive : ""}`}
							>
								<input
									type="radio"
									name="mode"
									value={option}
									checked={active}
									onChange={() => setMode(option)}
									className={styles.modeRadio}
									required
								/>
								<span className={styles.modeTitle}>{vocab.title}</span>
								<span className={styles.modeDescription}>
									{vocab.description}
								</span>
							</label>
						);
					})}
				</div>
			</fieldset>

			{/* 3. Options */}
			<div className={styles.optionRow}>
				<label className={styles.option} htmlFor={nameId}>
					<span className={styles.optionLabel}>Project name</span>
					<input
						id={nameId}
						className={styles.textInput}
						value={name}
						placeholder={video ? stem(video.name) : "clip_007"}
						disabled={disabled}
						onChange={(event) => {
							setName(event.target.value);
							setNameTouched(true);
						}}
					/>
				</label>
				<label className={styles.optionSmall} htmlFor={stepId}>
					<span className={styles.optionLabel}>Keep every</span>
					<span className={styles.stepGroup}>
						<input
							id={stepId}
							type="number"
							min={1}
							max={60}
							step={1}
							className={styles.numberInput}
							value={frameStep}
							disabled={disabled}
							onChange={(event) =>
								setFrameStep(
									Math.max(1, Math.min(60, Number(event.target.value) || 1)),
								)
							}
						/>
						<span className={styles.stepSuffix}>
							{frameStep === 1 ? "frame" : "th frame"}
						</span>
					</span>
				</label>
			</div>

			<div className={styles.actions}>
				<button
					type="submit"
					className="btn btnPrimary"
					disabled={!canSubmit}
					title={
						!video
							? "Choose a video first"
							: !mode
								? "Choose a segmentation mode"
								: undefined
					}
				>
					Create project
				</button>
				{mode && (
					<span className={styles.summary}>
						Will be created as a <strong>{MODE_VOCABULARY[mode].title}</strong>{" "}
						project
						{frameStep > 1 ? ` (every ${frameStep}th frame)` : ""}.
					</span>
				)}
			</div>
		</form>
	);
}

export function LockIcon({ size = 12 }: { size?: number }) {
	return (
		<svg
			width={size}
			height={size}
			viewBox="0 0 16 16"
			aria-hidden="true"
			focusable="false"
			className={styles.lockIcon}
		>
			<rect
				x="3"
				y="7"
				width="10"
				height="7"
				rx="1.5"
				fill="currentColor"
			/>
			<path
				d="M5 7V5a3 3 0 0 1 6 0v2"
				fill="none"
				stroke="currentColor"
				strokeWidth="1.6"
			/>
		</svg>
	);
}
