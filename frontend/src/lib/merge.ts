import type { RawRle, Tracklet } from "../types";
import { rleArea, rleIsEmpty, unionRle } from "./rle";

/**
 * Where a merge applies.
 *
 * `all` collapses two identities for good; `frame` and `range` only move masks
 * and leave both tracklets in place, which is the gesture for combining two
 * masks on one moment rather than fixing an identity split.
 */
export type MergeScope =
    | { kind: "all" }
    | { kind: "frame"; frame: number }
    | { kind: "range"; first: number; last: number };

/**
 * What happens on a frame both tracklets cover.
 *
 * `union` is for a body fragmented by an occluder; the two preferences are for
 * duplicate detections where one side is known to be the better mask.
 */
export type MergePolicy = "union" | "prefer-target" | "prefer-source";

export interface MergePlanFrame {
    frame: number;
    target: RawRle | null;
    source: RawRle | null;
    result: RawRle | null;
    /** Both tracklets had a mask on this frame. */
    conflict: boolean;
    /**
     * The merged mask is human-verified only when every pixel that survived
     * came from a human-drawn mask. Machine pixels must never be promoted.
     */
    human: boolean;
    /** The target's mask on this frame changes. */
    changed: boolean;
}

export interface MergePlan {
    targetId: number;
    /** Every source being absorbed. The UI passes one; the planner accepts N. */
    sourceIds: number[];
    /** Merge scope, inclusive and clamped to the clip. */
    first: number;
    last: number;
    policy: MergePolicy;
    labelId: number | null;
    /** Frames where at least one side has a mask. Empty frames are omitted. */
    frames: MergePlanFrame[];
    stats: {
        /** Frames only the target covers. */
        targetOnly: number;
        /** Frames only the source covers: the occlusion gap-fill case. */
        sourceOnly: number;
        /** Frames both cover. */
        conflicts: number;
        /** Frames whose target mask is written by this merge. */
        changed: number;
        /** Frames whose result is human-verified. */
        human: number;
        /** Frames whose result is machine-propagated. */
        propagated: number;
        /** Hand-verified target frames that become propagated. */
        downgraded: number;
        targetArea: number;
        sourceArea: number;
        resultArea: number;
    };
    /** Frames the target gains or changes, for dirty-frame tracking. */
    touchedFrames: number[];
    /** Sources with no masks outside the scope; they are removed entirely. */
    removedSourceIds: number[];
    /** Sources that keep masks outside the scope. */
    remainingSources: { id: number; frames: number }[];
}

interface PlanMergeParams {
    target: Tracklet;
    sources: Tracklet[];
    scope: MergeScope;
    policy: MergePolicy;
    labelId: number | null;
    frameCount: number;
    isHuman: (trackletId: number, frame: number) => boolean;
}

function maskAt(tracklet: Tracklet, frame: number): RawRle | null {
    const seg = tracklet.segmentations[frame];
    return seg && seg.counts && !rleIsEmpty(seg) ? seg : null;
}

function clampFrame(frame: number, frameCount: number): number {
    if (!Number.isFinite(frame)) return 0;
    return Math.max(
        0,
        Math.min(Math.max(0, frameCount - 1), Math.trunc(frame)),
    );
}

/**
 * Work out what a merge would do without changing anything.
 *
 * The dialog uses the result for its counts and the apply step turns it into a
 * new `Clip` in one update, so the reviewer sees the consequences before a
 * source tracklet disappears.
 */
export function planTrackletMerge({
    target,
    sources,
    scope,
    policy,
    labelId,
    frameCount,
    isHuman,
}: PlanMergeParams): MergePlan {
    // A source named twice, or named as the target, would be applied twice.
    const seen = new Set<number>([target.id]);
    const uniqueSources = sources.filter((source) => {
        if (seen.has(source.id)) return false;
        seen.add(source.id);
        return true;
    });

    let first = 0;
    let last = Math.max(0, frameCount - 1);
    if (scope.kind === "frame") {
        first = clampFrame(scope.frame, frameCount);
        last = first;
    } else if (scope.kind === "range") {
        first = clampFrame(Math.min(scope.first, scope.last), frameCount);
        last = clampFrame(Math.max(scope.first, scope.last), frameCount);
    }

    const frames: MergePlanFrame[] = [];
    const touched: number[] = [];
    let targetOnly = 0;
    let sourceOnly = 0;
    let conflicts = 0;
    let changed = 0;
    let human = 0;
    let propagated = 0;
    let downgraded = 0;
    let targetArea = 0;
    let sourceArea = 0;
    let resultArea = 0;

    for (let frame = first; frame <= last; frame++) {
        const targetMask = maskAt(target, frame);
        const contributingSources = uniqueSources.filter(
            (source) => maskAt(source, frame) !== null,
        );
        let sourceMask: RawRle | null = null;
        for (const source of contributingSources) {
            const mask = maskAt(source, frame);
            if (!mask) continue;
            sourceMask = sourceMask ? unionRle(sourceMask, mask) : mask;
        }
        if (!targetMask && !sourceMask) continue;

        const both = Boolean(targetMask && sourceMask);
        const targetHuman = targetMask ? isHuman(target.id, frame) : false;
        const sourceHuman =
            contributingSources.length > 0 &&
            contributingSources.every((source) => isHuman(source.id, frame));

        let result: RawRle | null;
        let resultHuman: boolean;
        if (targetMask && sourceMask) {
            if (policy === "prefer-target") {
                result = targetMask;
                resultHuman = targetHuman;
            } else if (policy === "prefer-source") {
                result = sourceMask;
                resultHuman = sourceHuman;
            } else {
                result = unionRle(targetMask, sourceMask);
                resultHuman = targetHuman && sourceHuman;
            }
        } else if (targetMask) {
            result = targetMask;
            resultHuman = targetHuman;
        } else {
            result = sourceMask;
            resultHuman = sourceHuman;
        }

        const frameChanged = targetMask
            ? both
                ? policy !== "prefer-target"
                : false
            : sourceMask !== null;
        const downgrade = targetHuman && !resultHuman;

        if (both) conflicts += 1;
        else if (targetMask) targetOnly += 1;
        else sourceOnly += 1;
        if (frameChanged) {
            changed += 1;
            touched.push(frame);
        }
        if (result) {
            const area = rleArea(result);
            resultArea += area;
            if (resultHuman) human += 1;
            else propagated += 1;
        }
        if (downgrade) downgraded += 1;
        targetArea += targetMask ? rleArea(targetMask) : 0;
        sourceArea += sourceMask ? rleArea(sourceMask) : 0;

        frames.push({
            frame,
            target: targetMask,
            source: sourceMask,
            result,
            conflict: both,
            human: resultHuman,
            changed: frameChanged,
        });
    }

    const removedSourceIds: number[] = [];
    const remainingSources: { id: number; frames: number }[] = [];
    for (const source of uniqueSources) {
        let outsideFrames = 0;
        for (let frame = 0; frame < frameCount; frame++) {
            if (frame >= first && frame <= last) continue;
            if (maskAt(source, frame)) outsideFrames += 1;
        }
        if (outsideFrames > 0) {
            remainingSources.push({ id: source.id, frames: outsideFrames });
        } else {
            removedSourceIds.push(source.id);
        }
    }

    return {
        targetId: target.id,
        sourceIds: uniqueSources.map((source) => source.id),
        first,
        last,
        policy,
        labelId,
        frames,
        stats: {
            targetOnly,
            sourceOnly,
            conflicts,
            changed,
            human,
            propagated,
            downgraded,
            targetArea,
            sourceArea,
            resultArea,
        },
        touchedFrames: touched,
        removedSourceIds,
        remainingSources,
    };
}
