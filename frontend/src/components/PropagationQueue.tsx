import type { JobState, PropagatedFrame } from "../lib/propagateApi";

/**
 * The propagation queue.
 *
 * Every object queued for propagation becomes its own job, and the backend works
 * through them one at a time — one tracker session, one object at a time — so
 * queueing ten objects costs the same GPU memory as queueing one. This renders
 * that list: what is waiting, what is running, and what each run produced.
 *
 * Only one run can be painted on the canvas at a time, so the row being watched
 * is marked and any other row with finished masks can be pulled up with Review.
 */

export interface PropQueueEntry {
    jobId: string;
    trackletId: number;
    label: string;
    color: string;
    /** The frame carrying the mask this run started from. */
    anchor: number;
    first: number;
    last: number;
    state: JobState;
    framesDone: number;
    framesTotal: number;
    elapsedMs: number;
    error: string | null;
    /**
     * How this run's result lands on the clip.
     *
     * `skip-existing` fills only the frames that have no mask, which is what a
     * first pass wants. `replace-range` overwrites the range, which is the only
     * thing that makes sense for a refinement: the frames being replaced are
     * stale masks produced by the run this one is correcting.
     */
    writePolicy: "skip-existing" | "replace-range";
    /** How many hand-verified frames seeded this run, the anchor excluded. */
    pinned: number;
    /**
     * Window boundaries of the run, as frame indices (the first window excluded).
     *
     * A long run is split into overlapping windows and the hand-off between two
     * of them is where a track can drift, so the timeline marks them.
     */
    windows: number[];
    /** Frames produced so far, keyed by frame index. */
    masks: Map<number, PropagatedFrame>;
}

const STATE_LABEL: Record<JobState, string> = {
    queued: "Waiting",
    running: "Running",
    done: "Done",
    cancelled: "Stopped",
    failed: "Failed",
};

const STATE_COLOR: Record<JobState, string> = {
    queued: "rgba(255, 204, 51, 0.95)",
    running: "#4aa3ff",
    done: "rgba(46, 204, 113, 0.95)",
    cancelled: "rgba(127, 127, 127, 0.95)",
    failed: "#e74c3c",
};

/** True while the run is still queued or going, i.e. not settled yet. */
export function isLive(entry: PropQueueEntry): boolean {
    return entry.state === "queued" || entry.state === "running";
}

interface PropagationQueueProps {
    entries: PropQueueEntry[];
    /** The run whose masks the canvas is showing, if any. */
    watchingId: string | null;
    onWatch: (jobId: string) => void;
    onCancel: (jobId: string) => void;
    onCancelAll: () => void;
}

export function PropagationQueue({
    entries,
    watchingId,
    onWatch,
    onCancel,
    onCancelAll,
}: PropagationQueueProps) {
    if (entries.length === 0) return null;
    const live = entries.filter(isLive).length;
    const ready = entries.filter(
        (entry) => entry.state === "done" && entry.masks.size > 0,
    ).length;

    return (
        <div
            style={{
                // A row in the video column, not a flexible pane: `0 0 auto` keeps
                // it to the height of its rows so the stage below keeps the space.
                // (Its old `1 1 100%` — copied from the in-bar range control, where
                // it meant "next line, full width" — made the queue grow to fill
                // the column and collapsed the video to nothing.)
                flex: "0 0 auto",
                // A long queue scrolls rather than shouldering the video aside.
                maxHeight: "30%",
                overflowY: "auto",
                display: "flex",
                flexDirection: "column",
                gap: 3,
                marginTop: 4,
                padding: "6px 8px",
                borderRadius: 4,
                border: "1px solid rgba(127, 127, 127, 0.3)",
                background: "rgba(127, 127, 127, 0.08)",
            }}
        >
            <div
                style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    fontSize: 12,
                    opacity: 0.85,
                }}
            >
                <strong style={{ fontSize: 12 }}>
                    Queue · {entries.length}
                </strong>
                {live > 0 && <span>{live} to go</span>}
                {ready > 0 && (
                    <span style={{ color: STATE_COLOR.done }}>
                        {ready} ready to review
                    </span>
                )}
                <span style={{ flex: "1 1 auto" }} />
                {live > 0 && (
                    <button
                        type="button"
                        className="btn"
                        onClick={onCancelAll}
                        title="Stop every run in the queue"
                    >
                        Stop queue
                    </button>
                )}
            </div>

            {entries.map((entry) => {
                const watching = entry.jobId === watchingId;
                const produced = entry.masks.size;
                const showProgress = isLive(entry) && entry.framesTotal > 0;
                return (
                    <div
                        key={entry.jobId}
                        style={{
                            display: "flex",
                            alignItems: "center",
                            gap: 8,
                            padding: "3px 5px",
                            borderRadius: 3,
                            fontSize: 12,
                            background: watching
                                ? "rgba(74, 163, 255, 0.12)"
                                : "transparent",
                        }}
                    >
                        <span
                            style={{
                                width: 9,
                                height: 9,
                                flex: "0 0 auto",
                                borderRadius: 2,
                                background: entry.color,
                            }}
                        />
                        <button
                            type="button"
                            className="btn"
                            style={{ padding: "1px 6px", fontSize: 12 }}
                            onClick={() => onWatch(entry.jobId)}
                            disabled={produced === 0}
                            title={
                                produced === 0
                                    ? "No frames produced yet"
                                    : "Show this run's masks on the canvas"
                            }
                        >
                            {entry.label} #{entry.trackletId}
                        </button>
                        <span
                            style={{
                                color: STATE_COLOR[entry.state],
                                fontSize: 11,
                                fontWeight: 600,
                            }}
                        >
                            {STATE_LABEL[entry.state]}
                        </span>
                        <span style={{ fontSize: 11, opacity: 0.75 }}>
                            {showProgress
                                ? `${entry.framesDone}/${entry.framesTotal} frames`
                                : `${produced} frame${produced === 1 ? "" : "s"}`}
                            {entry.elapsedMs > 0
                                ? ` · ${(entry.elapsedMs / 1000).toFixed(1)}s`
                                : ""}
                        </span>
                        {entry.error && (
                            <span style={{ fontSize: 11, color: "#e74c3c" }}>
                                {entry.error}
                            </span>
                        )}
                        <span style={{ flex: "1 1 auto" }} />
                        {watching && (
                            <span
                                style={{ fontSize: 10, opacity: 0.7 }}
                                title="These are the masks on the canvas"
                            >
                                on canvas
                            </span>
                        )}
                        {isLive(entry) && (
                            <button
                                type="button"
                                className="btn"
                                style={{ padding: "1px 6px", fontSize: 11 }}
                                onClick={() => onCancel(entry.jobId)}
                                title="Stop this run; frames already produced stay for review"
                            >
                                Stop
                            </button>
                        )}
                    </div>
                );
            })}
        </div>
    );
}
