import type { ReactNode } from "react";
import styles from "./Feedback.module.css";
import { Icon, type IconName } from "./Icon";

export interface ProgressBarProps {
    /** 0..1; omitted renders an indeterminate bar. */
    value?: number;
    label?: string;
    tone?: "accent" | "amber";
}

/** A thin determinate/indeterminate bar for exports and propagation runs. */
export function ProgressBar({
    value,
    label,
    tone = "accent",
}: ProgressBarProps) {
    const pct =
        value === undefined
            ? undefined
            : Math.round(Math.min(1, Math.max(0, value)) * 100);
    return (
        <div className={styles.progressWrap}>
            <div
                className={`${styles.track} ${tone === "amber" ? styles.amber : ""}`}
                role="progressbar"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={pct}
                aria-label={label}
            >
                <div
                    className={`${styles.fill} ${pct === undefined ? styles.indeterminate : ""}`}
                    style={pct === undefined ? undefined : { width: `${pct}%` }}
                />
            </div>
            {label ? (
                <span className={styles.progressLabel}>{label}</span>
            ) : null}
        </div>
    );
}

export interface EmptyStateProps {
    icon?: IconName;
    title: string;
    children?: ReactNode;
}

/** The placeholder shown when a region has nothing to display yet. */
export function EmptyState({ icon, title, children }: EmptyStateProps) {
    return (
        <div className={styles.empty}>
            {icon ? (
                <span className={styles.emptyIcon}>
                    <Icon name={icon} size={22} />
                </span>
            ) : null}
            <p className={styles.emptyTitle}>{title}</p>
            {children ? <p className={styles.emptyBody}>{children}</p> : null}
        </div>
    );
}
