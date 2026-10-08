import { useCallback, useEffect, useRef, useState } from "react";
import { Clip } from "./lib/clip";
import { SessionFrameSource, type FrameSource } from "./lib/frames";
import { closeProject, openProject } from "./lib/sessionsApi";
import { ZipArchive } from "./lib/zip";
import { UploadScreen } from "./components/UploadScreen";
import { Workspace, type WorkspaceNotice } from "./components/Workspace";
import { Button, Icon } from "./ui";
import styles from "./App.module.css";

/**
 * The shell around the two faces of the tool: the start screen (where
 * `CreateWizard` packs a project in the browser) and the review workspace.
 *
 * Opening a project does two things: the archive is parsed locally for its
 * annotation, and it is uploaded so the backend can materialise its frames. The
 * frames are what gets displayed, so an archive carrying only a video opens
 * exactly like one carrying a frame folder.
 */
type Phase = "upload" | "loading" | "ready" | "error";

function App() {
    const [phase, setPhase] = useState<Phase>("upload");
    const [clip, setClip] = useState<Clip | null>(null);
    const [zip, setZip] = useState<ZipArchive | null>(null);
    const [frames, setFrames] = useState<FrameSource | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [notice, setNotice] = useState<WorkspaceNotice | null>(null);

    // The session must be reachable from an unload handler, which cannot read
    // React state, so the id is mirrored in a ref.
    const sessionRef = useRef<string | null>(null);

    const releaseSession = useCallback(() => {
        const id = sessionRef.current;
        if (!id) return;
        sessionRef.current = null;
        closeProject(id);
    }, []);

    // A page that goes away without releasing its session leaves the frames for
    // the sweeper to collect; `keepalive` gives the request a chance to outlive
    // the page and avoid that wait.
    useEffect(() => {
        const onHide = () => releaseSession();
        window.addEventListener("pagehide", onHide);
        return () => {
            window.removeEventListener("pagehide", onHide);
            releaseSession();
        };
    }, [releaseSession]);

    const openArchive = useCallback(
        async (file: File) => {
            const archive = await ZipArchive.fromFile(file);
            const opened = await openProject(file);
            sessionRef.current = opened.sessionId;
            try {
                // The session measured the frames it produced, so it describes
                // the clip; the archive still supplies the annotations.
                const parsed = await Clip.fromZip(archive, {
                    frames: {
                        frameNames: opened.frameNames,
                        fps: opened.fps,
                        width: opened.width,
                        height: opened.height,
                        external: true,
                    },
                });
                setZip(archive);
                setFrames(
                    new SessionFrameSource(opened.sessionId, opened.frameNames),
                );
                setClip(parsed);
                setNotice(
                    opened.frameNamesMatch
                        ? null
                        : {
                              text:
                                  "The frames the backend prepared do not match " +
                                  "the list this archive recorded (" +
                                  `${opened.recordedFrameNames.length} recorded, ` +
                                  `${opened.frameCount} built). Masks are addressed ` +
                                  "by frame position, so check before annotating.",
                          },
                );
                setPhase("ready");
            } catch (cause) {
                // An unusable session should not linger on disk.
                releaseSession();
                throw cause;
            }
        },
        [releaseSession],
    );

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
        releaseSession();
        setClip(null);
        setZip(null);
        setFrames(null);
        setError(null);
        setNotice(null);
        setPhase("upload");
    }, [releaseSession]);

    return (
        <div className={styles.app}>
            {phase === "upload" && <UploadScreen onFile={handleFile} />}

            {phase === "loading" && (
                <div className={styles.center}>
                    <span className={styles.brand}>
                        <Icon name="leaf" size={24} />
                    </span>
                    <div className={styles.spinner} />
                    <p className={styles.subtitle}>
                        Preparing frames and reading annotations…
                    </p>
                </div>
            )}

            {phase === "error" && (
                <div className={styles.center}>
                    <div className={styles.errorCard}>
                        <h1 className={styles.errorTitle}>
                            Could not open project
                        </h1>
                        <p className={styles.error}>{error}</p>
                        <Button variant="primary" onClick={handleReset}>
                            Back to start
                        </Button>
                    </div>
                </div>
            )}

            {phase === "ready" &&
                clip !== null &&
                zip !== null &&
                frames !== null && (
                    <Workspace
                        key={clip.name}
                        clip={clip}
                        zip={zip}
                        frames={frames}
                        notice={notice}
                        onDismissNotice={() => setNotice(null)}
                        onReset={handleReset}
                    />
                )}
        </div>
    );
}

export default App;
