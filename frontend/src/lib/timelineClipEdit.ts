export interface TimelineClipRangeEdit {
    originalFirst: number;
    originalLast: number;
    first: number;
    last: number;
    deleted: boolean;
    removeRange: { first: number; last: number } | null;
}

/** Frame indices removed by a staged edit, with inclusive range endpoints. */
export function removedTimelineFrames(edit: TimelineClipRangeEdit): number[] {
    const removed: number[] = [];
    for (let frame = edit.originalFirst; frame <= edit.originalLast; frame++) {
        if (
            edit.deleted ||
            frame < edit.first ||
            frame > edit.last ||
            (edit.removeRange !== null &&
                frame >= edit.removeRange.first &&
                frame <= edit.removeRange.last)
        )
            removed.push(frame);
    }
    return removed;
}
