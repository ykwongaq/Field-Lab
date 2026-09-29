import { useRef, useState } from "react";
import { CreateWizard } from "./CreateWizard";
import styles from "./UploadScreen.module.css";

interface UploadScreenProps {
    onFile: (file: File) => void;
}

export function UploadScreen({ onFile }: UploadScreenProps) {
    const inputRef = useRef<HTMLInputElement>(null);
    const [dragOver, setDragOver] = useState(false);

    return (
        <div className={styles.wrap}>
            <div className={styles.inner}>
                <header className={styles.intro}>
                    <h1 className={styles.title}>
                        Video Segmentation Reviewer
                    </h1>
                    <p className={styles.subtitle}>
                        Create a project from a video or a folder of frames, or
                        open one you already have
                    </p>
                </header>

                <CreateWizard onOpen={onFile} />

                <div
                    role="button"
                    tabIndex={0}
                    className={`${styles.existing} ${dragOver ? styles.dragOver : ""}`}
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
                        Drop a <code>.project</code> archive here (older{" "}
                        <code>.zip</code> archives work too), or click to
                        browse.
                    </p>
                    <input
                        ref={inputRef}
                        type="file"
                        accept=".project,.zip,application/zip"
                        className={styles.input}
                        onChange={(event) => {
                            const file = event.target.files?.[0];
                            if (file) onFile(file);
                            event.target.value = "";
                        }}
                    />
                </div>
            </div>
        </div>
    );
}
