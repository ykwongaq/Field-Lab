/** The little padlock that flags a decision that cannot be undone. */
export function LockIcon({ size = 12 }: { size?: number }) {
    return (
        <svg
            width={size}
            height={size}
            viewBox="0 0 16 16"
            aria-hidden="true"
            focusable="false"
            style={{ flex: "0 0 auto" }}
        >
            <rect
                x="3"
                y="7"
                width="10"
                height="7"
                rx="1.5"
                fill="currentColor"
            />
            <path
                d="M5 7V5a3 3 0 0 1 6 0v2"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.6"
            />
        </svg>
    );
}
