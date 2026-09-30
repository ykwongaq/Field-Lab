import { useEffect } from "react";
import styles from "./ExportMenu.module.css";

/**
 * One thing the workspace can hand back.
 *
 * `detail` describes what the download will contain; `disabledReason`, when
 * present, says why it cannot be exported at all and replaces the detail.
 */
export interface ExportOption {
    id: string;
    title: string;
    detail: string;
    disabledReason?: string;
    run: () => void | Promise<void>;
}

export interface ExportMenuProps {
    title: string;
    options: readonly ExportOption[];
    /** The option currently being written, or `null` when nothing is running. */
    busyId: string | null;
    /** Progress read-out for the running option ("12 / 340 frames"). */
    busyText?: string | null;
    error?: string | null;
    onClose: () => void;
}

/**
 * The chooser behind the Export button.
 *
 * Deliberately a modal rather than a menu: each entry downloads its own file,
 * and a running export can take a while on a large clip, so the list stays open
 * (with the other choices disabled) until the file is handed to the browser.
 */
export function ExportMenu({
    title,
    options,
    busyId,
    busyText,
    error = null,
    onClose,
}: ExportMenuProps) {
    // Escape closes. While an export runs the menu stays put, so a stray key
    // press cannot dismiss the only progress read-out.
    useEffect(() => {
        const onKey = (event: KeyboardEvent) => {
            if (event.key === "Escape" && busyId === null) onClose();
        };
        window.addEventListener("keydown", onKey);
        return () => window.removeEventListener("keydown", onKey);
    }, [busyId, onClose]);

    return (
        <div
            className={styles.backdrop}
            role="presentation"
            onMouseDown={(event) => {
                if (event.target === event.currentTarget && busyId === null)
                    onClose();
            }}
        >
            <div
                className={styles.dialog}
                role="dialog"
                aria-modal="true"
                aria-label={title}
            >
                <header className={styles.head}>
                    <h2 className={styles.title}>{title}</h2>
                    <button
                        type="button"
                        className={styles.close}
                        onClick={onClose}
                        disabled={busyId !== null}
                        aria-label="Close"
                    >
                        ×
                    </button>
                </header>
                <p className={styles.lede}>
                    Each choice saves its own file. Nothing is re-encoded — the
                    bytes are copied from this project.
                </p>

                <ul className={styles.list}>
                    {options.map((option) => {
                        const busy = busyId === option.id;
                        const blocked = option.disabledReason !== undefined;
                        const note = busy
                            ? (busyText ?? "Working…")
                            : (option.disabledReason ?? option.detail);
                        return (
                            <li key={option.id} className={styles.item}>
                                <div className={styles.text}>
                                    <span className={styles.itemTitle}>
                                        {option.title}
                                    </span>
                                    <span
                                        className={`${styles.itemDetail} ${
                                            blocked ? styles.itemBlocked : ""
                                        }`}
                                    >
                                        {note}
                                    </span>
                                </div>
                                <button
                                    type="button"
                                    className="btn"
                                    onClick={() => void option.run()}
                                    disabled={blocked || busyId !== null}
                                >
                                    {busy ? "Exporting…" : "Export"}
                                </button>
                            </li>
                        );
                    })}
                </ul>

                {error && (
                    <p className={styles.error} role="alert">
                        {error}
                    </p>
                )}
            </div>
        </div>
    );
}
