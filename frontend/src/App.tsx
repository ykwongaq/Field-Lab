import { useCallback, useState } from "react";
import { Clip } from "./lib/clip";
import { ZipArchive } from "./lib/zip";
import { UploadScreen } from "./components/UploadScreen";
import { Workspace, type WorkspaceNotice } from "./components/Workspace";
import styles from "./App.module.css";

/**
 * The shell around the two faces of the tool: the start screen (where
 * `CreateWizard` packs a project in the browser) and the review workspace.
 * Creation itself lives in the wizard; this only opens finished archives.
 */
type Phase = "upload" | "loading" | "ready" | "error";

function App() {
    const [phase, setPhase] = useState<Phase>("upload");
    const [clip, setClip] = useState<Clip | null>(null);
    const [zip, setZip] = useState<ZipArchive | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [notice, setNotice] = useState<WorkspaceNotice | null>(null);

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
                setError(
                    cause instanceof Error ? cause.message : String(cause),
                );
                setPhase("error");
            }
        },
        [openArchive],
    );

    const handleReset = useCallback(() => {
        setClip(null);
        setZip(null);
        setError(null);
        setNotice(null);
        setPhase("upload");
    }, []);

    return (
        <div className={styles.app}>
            {phase === "upload" && <UploadScreen onFile={handleFile} />}

            {phase === "loading" && (
                <div className={styles.center}>
                    <h1 className={styles.title}>Loading</h1>
                    <p className={styles.subtitle}>
                        Reading frames and annotations…
                    </p>
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
