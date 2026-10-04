import type { ReactNode } from "react";
import { Icon, type IconName } from "./Icon";

export type ChipTone = "neutral" | "accent" | "amber" | "good" | "warn" | "bad";

const TONE_CLASS: Record<ChipTone, string> = {
    neutral: "",
    accent: "chipAccent",
    amber: "chipAmber",
    good: "chipGood",
    warn: "chipWarn",
    bad: "chipBad",
};

export interface ChipProps {
    children: ReactNode;
    tone?: ChipTone;
    icon?: IconName;
    title?: string;
}

/** A compact status pill — mode badges, counts, and inline state markers. */
export function Chip({ children, tone = "neutral", icon, title }: ChipProps) {
    const classes = ["chip"];
    if (TONE_CLASS[tone]) classes.push(TONE_CLASS[tone]);
    return (
        <span className={classes.join(" ")} title={title}>
            {icon ? <Icon name={icon} size={12} /> : null}
            {children}
        </span>
    );
}

export interface SegmentedOption<T extends string> {
    value: T;
    label: string;
    title?: string;
    disabled?: boolean;
}

export interface SegmentedProps<T extends string> {
    options: readonly SegmentedOption<T>[];
    value: T;
    onChange: (value: T) => void;
    ariaLabel: string;
}

/**
 * A radio group drawn as a single pill of joined buttons — used for the drawing
 * method, paint mode and other small either/or choices.
 */
export function Segmented<T extends string>({
    options,
    value,
    onChange,
    ariaLabel,
}: SegmentedProps<T>) {
    return (
        <div className="segmented" role="radiogroup" aria-label={ariaLabel}>
            {options.map((option) => (
                <button
                    key={option.value}
                    type="button"
                    role="radio"
                    aria-checked={option.value === value}
                    disabled={option.disabled}
                    title={option.title}
                    className={`segment${
                        option.value === value ? " segmentActive" : ""
                    }`}
                    onClick={() => onChange(option.value)}
                >
                    {option.label}
                </button>
            ))}
        </div>
    );
}
