import { useRef, useState } from "react";
import {
	CreateProjectPanel,
	type CreateProjectRequest,
} from "./CreateProjectPanel";
import styles from "./UploadScreen.module.css";

interface UploadScreenProps {
	onFile: (file: File) => void;
	onCreate: (request: CreateProjectRequest) => void;
}

export function UploadScreen({ onFile, onCreate }: UploadScreenProps) {
	const inputRef = useRef<HTMLInputElement>(null);
	const [dragOver, setDragOver] = useState(false);

	return (
		<div className={styles.wrap}>
			<div className={styles.inner}>
				<header className={styles.intro}>
					<h1 className={styles.title}>Video Segmentation Reviewer</h1>
					<p className={styles.subtitle}>
						Verify tracklet masks and taxonomic labels
					</p>
				</header>

				<div className={styles.columns}>
					<div
						role="button"
						tabIndex={0}
						className={`${styles.dropzone} ${dragOver ? styles.dragOver : ""}`}
						onClick={() => inputRef.current?.click()}
						onKeyDown={(event) => {
							if (event.key === "Enter" || event.key === " ") {
								event.preventDefault();
								inputRef.current?.click();
							}
						}}
						onDragOver={(event) => {
							event.preventDefault();
							setDragOver(true);
						}}
						onDragLeave={() => setDragOver(false)}
						onDrop={(event) => {
							event.preventDefault();
							setDragOver(false);
							const file = event.dataTransfer.files?.[0];
							if (file) onFile(file);
						}}
					>
						<h2 className={styles.heading}>Open an existing project</h2>
						<p className={styles.hint}>
							Drop a project <code>.zip</code> archive here, or click to
							browse.
						</p>
						<p className={styles.detail}>
							The archive contains <code>frames/</code> and an annotation JSON
							under <code>annotations/</code>; its video record carries the
							locked <code>segmentation_mode</code>. Older archives without
							that field open as instance projects.
						</p>
						<input
							ref={inputRef}
							type="file"
							accept=".zip,application/zip"
							className={styles.input}
							onChange={(event) => {
								const file = event.target.files?.[0];
								if (file) onFile(file);
								event.target.value = "";
							}}
						/>
					</div>

					<div className={styles.divider} aria-hidden="true">
						<span>or</span>
					</div>

					<CreateProjectPanel onCreate={onCreate} />
				</div>
			</div>
		</div>
	);
}