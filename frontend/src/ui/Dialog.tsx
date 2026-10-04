import { useEffect, useRef, type ReactNode } from "react";
import { Icon } from "./Icon";
import styles from "./Dialog.module.css";

export interface DialogProps {
    title: string;
    onClose: () => void;
    children: ReactNode;
    /** Wider variant for content-heavy dialogs. */
    wide?: boolean;
    /** Footer content pinned below a scrollable body. */
    footer?: ReactNode;
    /** Blocks dismissal while a long-running action is in flight. */
    busy?: boolean;
}

/**
 * The one modal shell in the app.
 *
 * Closes on Escape and on a backdrop click, traps initial focus inside, and
 * refuses to dismiss while `busy` so a running job cannot be lost to a stray
 * key press.
 */
export function Dialog({
    title,
    onClose,
    children,
    wide = false,
    footer,
    busy = false,
}: DialogProps) {
    const panelRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const onKey = (event: KeyboardEvent) => {
            if (event.key === "Escape" && !busy) onClose();
        };
        window.addEventListener("keydown", onKey);
        return () => window.removeEventListener("keydown", onKey);
    }, [busy, onClose]);

    useEffect(() => {
        // Move focus into the dialog so keyboard users are not left behind the
        // backdrop; the close button is a safe, always-present target.
        panelRef.current
            ?.querySelector<HTMLElement>("[data-autofocus]")
            ?.focus();
    }, []);

    return (
        <div
            className={styles.backdrop}
            role="presentation"
            onMouseDown={(event) => {
                if (event.target === event.currentTarget && !busy) onClose();
            }}
        >
            <div
                ref={panelRef}
                className={`${styles.dialog} ${wide ? styles.wide : ""}`}
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
                        disabled={busy}
                        aria-label="Close"
                        data-autofocus
                    >
                        <Icon name="close" size={16} />
                    </button>
                </header>
                <div className={styles.body}>{children}</div>
                {footer ? (
                    <footer className={styles.foot}>{footer}</footer>
                ) : null}
            </div>
        </div>
    );
}
