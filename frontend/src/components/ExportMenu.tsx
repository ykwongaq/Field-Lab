import { Dialog, Icon } from "../ui";
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
 * Deliberately a dialog rather than a menu: each entry downloads its own file,
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
    return (
        <Dialog
            title={title}
            onClose={onClose}
            busy={busyId !== null}
            wide
            footer={
                <p className={styles.footNote}>
                    Files are written from this project — nothing is uploaded.
                </p>
            }
        >
            <ul className={styles.list}>
                {options.map((option) => {
                    const busy = busyId === option.id;
                    const blocked = option.disabledReason !== undefined;
                    return (
                        <li key={option.id} className={styles.item}>
                            <span className={styles.icon}>
                                <Icon name="download" size={16} />
                            </span>
                            <div className={styles.text}>
                                <span className={styles.itemTitle}>
                                    {option.title}
                                </span>
                                <span
                                    className={`${styles.itemDetail} ${
                                        blocked ? styles.itemBlocked : ""
                                    }`}
                                >
                                    {busy
                                        ? (busyText ?? "Working…")
                                        : (option.disabledReason ??
                                          option.detail)}
                                </span>
                            </div>
                            <button
                                type="button"
                                className="btn btnPrimary"
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
        </Dialog>
    );
}
