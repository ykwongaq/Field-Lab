import type { ButtonHTMLAttributes, ReactNode } from "react";
import { Icon, type IconName } from "./Icon";

export type ButtonVariant = "default" | "primary" | "ghost" | "danger";
export type ButtonSize = "md" | "sm";

const VARIANT_CLASS: Record<ButtonVariant, string> = {
    default: "",
    primary: "btnPrimary",
    ghost: "btnGhost",
    danger: "btnDanger",
};

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: ButtonVariant;
    size?: ButtonSize;
    /** Leading icon, and the whole label when `children` is omitted. */
    icon?: IconName;
    /** Trailing icon, e.g. an arrow on a "next" action. */
    iconRight?: IconName;
    children?: ReactNode;
}

/**
 * The button for every action in the app.
 *
 * Composes the shared `.btn` class family so variant styling has exactly one
 * definition; `icon` alone renders a square icon button with an accessible name
 * taken from `title` or `aria-label`.
 */
export function Button({
    variant = "default",
    size = "md",
    icon,
    iconRight,
    children,
    className,
    type = "button",
    ...rest
}: ButtonProps) {
    const classes = ["btn"];
    if (VARIANT_CLASS[variant]) classes.push(VARIANT_CLASS[variant]);
    if (size === "sm") classes.push("btnSmall");
    if (icon && !children) classes.push("btnIcon");
    if (className) classes.push(className);

    return (
        <button type={type} className={classes.join(" ")} {...rest}>
            {icon ? <Icon name={icon} size={size === "sm" ? 14 : 15} /> : null}
            {children}
            {iconRight ? (
                <Icon name={iconRight} size={size === "sm" ? 14 : 15} />
            ) : null}
        </button>
    );
}
