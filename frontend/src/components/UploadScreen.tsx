import { useRef, useState } from "react";
import { CreateWizard } from "./CreateWizard";
import { Icon } from "../ui";
import styles from "./UploadScreen.module.css";

interface UploadScreenProps {
    onFile: (file: File) => void;
}

/**
 * The landing screen: two ways in, side by side.
 *
 * The left pane creates a project (packed in the browser by `CreateWizard`);
 * the right pane opens one that already exists. Both panes share the viewport
 * height — neither scrolls the page, so the whole app stays one screen.
 */
export function UploadScreen({ onFile }: UploadScreenProps) {
    const inputRef = useRef<HTMLInputElement>(null);
    const [dragOver, setDragOver] = useState(false);

    const open = (file: File) => onFile(file);

    return (
        <div className={styles.landing}>
            <header className={styles.topbar}>
                <span className={styles.brand}>
                    <Icon name="leaf" size={18} />
                </span>
                <div>
                    <h1 className={styles.wordmark}>Field Lab</h1>
                    <p className={styles.tagline}>
                        Animal tracking &amp; video mask annotation
                    </p>
                </div>
            </header>

            <main className={styles.main}>
                <div className={styles.createPane}>
                    <CreateWizard onOpen={onFile} />
                </div>

                <aside className={styles.openPane}>
                    <div
                        role="button"
                        tabIndex={0}
                        className={`${styles.dropZone} ${
                            dragOver ? styles.dragOver : ""
                        }`}
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
                            if (file) open(file);
                        }}
                    >
                        <span className={styles.dropIcon}>
                            <Icon name="upload" size={22} />
                        </span>
                        <h2 className={styles.dropTitle}>Open a project</h2>
                        <p className={styles.dropHint}>
                            Drop a <code>.project</code> archive here, or click
                            to browse.
                        </p>
                        <input
                            ref={inputRef}
                            type="file"
                            accept=".project,.zip,application/zip"
                            className={styles.hiddenInput}
                            onChange={(event) => {
                                const file = event.target.files?.[0];
                                if (file) open(file);
                                event.target.value = "";
                            }}
                        />
                    </div>

                    <ul className={styles.tips}>
                        <li className={styles.tip}>
                            <Icon name="sparkles" size={15} />
                            <span>
                                Click, box, type or paint to segment with
                                SAM&nbsp;3.
                            </span>
                        </li>
                        <li className={styles.tip}>
                            <Icon name="propagate" size={15} />
                            <span>
                                Track a mask across frames, then correct and
                                re-run.
                            </span>
                        </li>
                        <li className={styles.tip}>
                            <Icon name="target" size={15} />
                            <span>
                                Label each animal with its full taxonomy.
                            </span>
                        </li>
                    </ul>
                </aside>
            </main>
        </div>
    );
}
