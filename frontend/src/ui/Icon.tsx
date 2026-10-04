import type { ReactNode } from "react";
import styles from "./Icon.module.css";

/**
 * The app's single icon set.
 *
 * Icons are stroke-only outlines on a 24×24 grid, so they inherit the current
 * text colour and weight consistently wherever they appear. Components should
 * never hand-roll their own `<svg>` — adding a path here keeps the visual
 * language in one place.
 */
export type IconName =
    | "select"
    | "plus"
    | "pencil"
    | "trash"
    | "propagate"
    | "play"
    | "pause"
    | "stepBack"
    | "stepForward"
    | "zoomIn"
    | "zoomOut"
    | "fit"
    | "eye"
    | "eyeOff"
    | "layers"
    | "search"
    | "chevronDown"
    | "close"
    | "check"
    | "info"
    | "alert"
    | "upload"
    | "folder"
    | "film"
    | "download"
    | "lock"
    | "target"
    | "brush"
    | "box"
    | "text"
    | "polygon"
    | "sparkles"
    | "leaf"
    | "sliders"
    | "undo"
    | "crosshair"
    | "refresh"
    | "arrowRight"
    | "minus";

const PATHS: Record<IconName, ReactNode> = {
    select: <path d="M5 3v15.5l4.2-4.2H16L5 3z" />,
    plus: <path d="M12 5v14M5 12h14" />,
    minus: <path d="M5 12h14" />,
    pencil: (
        <>
            <path d="M4 20h4L19 9l-4-4L4 16v4z" />
            <path d="M14 5l4 4" />
        </>
    ),
    trash: (
        <>
            <path d="M4 7h16M9 7V4h6v3" />
            <path d="M6 7l1 13h10l1-13" />
            <path d="M10 11v6M14 11v6" />
        </>
    ),
    propagate: (
        <>
            <path d="M4 12h10" />
            <path d="M10 8l4 4-4 4" />
            <path d="M17 8l4 4-4 4" />
        </>
    ),
    play: <path d="M8 5l11 7-11 7V5z" />,
    pause: <path d="M9 5v14M15 5v14" />,
    stepBack: (
        <>
            <path d="M17 6l-6 6 6 6" />
            <path d="M7 6v12" />
        </>
    ),
    stepForward: (
        <>
            <path d="M7 6l6 6-6 6" />
            <path d="M17 6v12" />
        </>
    ),
    zoomIn: (
        <>
            <circle cx="11" cy="11" r="7" />
            <path d="M20 20l-4.2-4.2M11 8.5v5M8.5 11h5" />
        </>
    ),
    zoomOut: (
        <>
            <circle cx="11" cy="11" r="7" />
            <path d="M20 20l-4.2-4.2M8.5 11h5" />
        </>
    ),
    fit: <path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" />,
    eye: (
        <>
            <path d="M2.5 12S6 6 12 6s9.5 6 9.5 6-3.5 6-9.5 6-9.5-6-9.5-6z" />
            <circle cx="12" cy="12" r="3" />
        </>
    ),
    eyeOff: (
        <>
            <path d="M4 4l16 16" />
            <path d="M9.9 5.2A9.6 9.6 0 0112 5c6 0 9.5 7 9.5 7a17 17 0 01-3 3.9M6.7 7.3A16.6 16.6 0 002.5 12S6 19 12 19a9.4 9.4 0 004.3-1" />
        </>
    ),
    layers: (
        <>
            <path d="M12 3l9 5-9 5-9-5 9-5z" />
            <path d="M3 12.5l9 5 9-5" />
        </>
    ),
    search: (
        <>
            <circle cx="11" cy="11" r="7" />
            <path d="M20 20l-4.2-4.2" />
        </>
    ),
    chevronDown: <path d="M6 9.5l6 6 6-6" />,
    close: <path d="M6 6l12 12M18 6L6 18" />,
    check: <path d="M4.5 12.5l5 5 10-11" />,
    info: (
        <>
            <circle cx="12" cy="12" r="9" />
            <path d="M12 11v5.5M12 7.6v.01" />
        </>
    ),
    alert: (
        <>
            <path d="M12 3l9.5 17h-19L12 3z" />
            <path d="M12 10v4.2M12 17.2v.01" />
        </>
    ),
    upload: (
        <>
            <path d="M12 16V4" />
            <path d="M7.5 8.5L12 4l4.5 4.5" />
            <path d="M4 20h16" />
        </>
    ),
    folder: (
        <path d="M3 6.5A1.5 1.5 0 014.5 5h4L10.5 7H19a1.5 1.5 0 011.5 1.5v9A1.5 1.5 0 0119 19H4.5A1.5 1.5 0 013 17.5v-11z" />
    ),
    film: (
        <>
            <rect x="3" y="4.5" width="18" height="15" rx="2" />
            <path d="M3 9.5h18M3 14.5h18M8 4.5v15M16 4.5v15" />
        </>
    ),
    download: (
        <>
            <path d="M12 4v12" />
            <path d="M7.5 11.5L12 16l4.5-4.5" />
            <path d="M4 20h16" />
        </>
    ),
    lock: (
        <>
            <rect x="5" y="10.5" width="14" height="9.5" rx="2" />
            <path d="M8.5 10.5V8a3.5 3.5 0 017 0v2.5" />
        </>
    ),
    target: (
        <>
            <circle cx="12" cy="12" r="8.5" />
            <circle cx="12" cy="12" r="3.5" />
        </>
    ),
    brush: (
        <>
            <path d="M14.5 4.8l4.7 4.7L9.4 19.3H4.7v-4.7L14.5 4.8z" />
            <path d="M12.4 6.9l4.7 4.7" />
        </>
    ),
    box: <rect x="4" y="6" width="16" height="12" rx="1.5" />,
    text: <path d="M5 7V5h14v2M12 5v14M9.5 19h5" />,
    polygon: <path d="M12 3.5l8 6-3 9.5H7l-3-9.5 8-6z" />,
    sparkles: (
        <>
            <path d="M11 3.5l1.7 4.4 4.4 1.7-4.4 1.7L11 15.7 9.3 11.3 4.9 9.6l4.4-1.7L11 3.5z" />
            <path d="M18.5 15l.9 2.2 2.2.9-2.2.9-.9 2.2-.9-2.2-2.2-.9 2.2-.9.9-2.2z" />
        </>
    ),
    leaf: (
        <>
            <path d="M20 4C11 4 4 8 4 15a5 5 0 005 5c7 0 11-7 11-16z" />
            <path d="M9.5 15.5c1.8-3.2 4.5-5.6 8-7" />
        </>
    ),
    sliders: (
        <path d="M4 7.5h9M17 7.5h3M4 16.5h3M11 16.5h9M15 4.5v6M8 13.5v6" />
    ),
    undo: (
        <>
            <path d="M8.5 8L4.5 12l4 4" />
            <path d="M4.5 12h9.5a5 5 0 010 10h-1.5" />
        </>
    ),
    crosshair: (
        <>
            <circle cx="12" cy="12" r="7.5" />
            <path d="M12 2.5v4M12 17.5v4M2.5 12h4M17.5 12h4" />
        </>
    ),
    refresh: (
        <>
            <path d="M20 12a8 8 0 11-2.4-5.7" />
            <path d="M20 4.5V10h-5.5" />
        </>
    ),
    arrowRight: (
        <>
            <path d="M4 12h14" />
            <path d="M13 6.5l5.5 5.5L13 17.5" />
        </>
    ),
};

export interface IconProps {
    name: IconName;
    /** Rendered size in pixels (the glyph is square). */
    size?: number;
    className?: string;
    /** Set when the icon is the only content of a control that needs a name. */
    title?: string;
}

export function Icon({ name, size = 18, className, title }: IconProps) {
    return (
        <svg
            className={className ? `${styles.icon} ${className}` : styles.icon}
            viewBox="0 0 24 24"
            width={size}
            height={size}
            aria-hidden={title ? undefined : true}
            role={title ? "img" : undefined}
            focusable="false"
        >
            {title ? <title>{title}</title> : null}
            {PATHS[name]}
        </svg>
    );
}
