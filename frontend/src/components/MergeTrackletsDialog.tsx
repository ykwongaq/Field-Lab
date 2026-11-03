import { useMemo, useState } from "react";
import type { Clip } from "../lib/clip";
import type { Tracklet } from "../types";
import {
    planTrackletMerge,
    type MergePlan,
    type MergePolicy,
    type MergeScope,
} from "../lib/merge";
import { MODE_VOCABULARY } from "../lib/project";
import { Button, Dialog } from "../ui";
import styles from "./MergeTrackletsDialog.module.css";

type ScopeKind = "all" | "frame" | "range";

interface MergeTrackletsDialogProps {
    clip: Clip;
    /** The identity that survives. */
    target: Tracklet;
    /** Every other selected tracklet, absorbed into the target. */
    sources: Tracklet[];
    currentFrame: number;
    isHuman: (trackletId: number, frame: number) => boolean;
    onTargetChange: (id: number) => void;
    onCancel: () => void;
    onConfirm: (plan: MergePlan) => void;
}

/**
 * Confirm a tracklet merge before the source identity disappears.
 *
 * The dialog owns the choices that change what "merge" means — scope, overlap
 * policy and label — and shows the resulting frame counts before anything is
 * written. Confirming is deliberately unavailable while overlapping frames
 * exist and no policy has been chosen: that conflict is the one case where a
 * silent union can corrupt a mask.
 */
export function MergeTrackletsDialog({
    clip,
    target,
    sources,
    currentFrame,
    isHuman,
    onTargetChange,
    onCancel,
    onConfirm,
}: MergeTrackletsDialogProps) {
    const vocab = MODE_VOCABULARY[clip.mode];
    const multi = sources.length > 1;
    const singleSource = sources.length === 1 ? sources[0] : null;
    const allTracklets = [target, ...sources];
    const firsts = allTracklets
        .map((tracklet) => tracklet.maskFrames.first)
        .filter((frame) => frame >= 0);
    const lasts = allTracklets
        .map((tracklet) => tracklet.maskFrames.last)
        .filter((frame) => frame >= 0);
    const defaultFirst = firsts.length > 0 ? Math.min(...firsts) : 0;
    const defaultLast =
        lasts.length > 0
            ? Math.max(...lasts)
            : Math.max(0, clip.frameCount - 1);

    const [scopeKind, setScopeKind] = useState<ScopeKind>("all");
    const [rangeFirst, setRangeFirst] = useState(defaultFirst + 1);
    const [rangeLast, setRangeLast] = useState(defaultLast + 1);
    const [policy, setPolicy] = useState<MergePolicy | null>(null);
    const labelsDiffer = sources.some(
        (source) => source.labelId !== target.labelId,
    );
    const [labelChoice, setLabelChoice] = useState<number | null>(
        target.labelId ??
            sources.find((source) => source.labelId !== null)?.labelId ??
            null,
    );

    const rangeInvalid =
        !multi &&
        scopeKind === "range" &&
        (!Number.isFinite(rangeFirst) ||
            !Number.isFinite(rangeLast) ||
            rangeFirst < 1 ||
            rangeLast > clip.frameCount ||
            rangeFirst > rangeLast);

    const scope = useMemo<MergeScope>(() => {
        // Three or more identities means their whole histories; a frame/range
        // merge only makes sense for a pair.
        if (multi) return { kind: "all" };
        if (scopeKind === "frame") {
            return { kind: "frame", frame: currentFrame };
        }
        if (scopeKind === "range") {
            return {
                kind: "range",
                first: rangeFirst - 1,
                last: rangeLast - 1,
            };
        }
        return { kind: "all" };
    }, [multi, scopeKind, currentFrame, rangeFirst, rangeLast]);

    const effectivePolicy: MergePolicy = policy ?? "union";
    const labelId = labelsDiffer ? labelChoice : target.labelId;
    const plan = useMemo(
        () =>
            planTrackletMerge({
                target,
                sources,
                scope,
                policy: effectivePolicy,
                labelId,
                frameCount: clip.frameCount,
                isHuman,
            }),
        [
            target,
            sources,
            scope,
            effectivePolicy,
            labelId,
            clip.frameCount,
            isHuman,
        ],
    );

    const needsPolicy = plan.stats.conflicts > 0 && policy === null;
    const labelChanged = labelsDiffer && labelChoice !== target.labelId;
    const canMerge =
        !rangeInvalid &&
        !needsPolicy &&
        (plan.stats.changed > 0 ||
            plan.removedSourceIds.length > 0 ||
            labelChanged);
    const resultFrames =
        plan.stats.targetOnly + plan.stats.sourceOnly + plan.stats.conflicts;
    const removedSources = plan.removedSourceIds;
    const keptSources = plan.remainingSources;

    const policyOptions: [MergePolicy, string][] = [
        ["union", multi ? "Combine all" : "Combine both"],
        ["prefer-target", `Keep #${target.id}`],
        ...(singleSource
            ? ([["prefer-source", `Keep #${singleSource.id}`]] as [
                  MergePolicy,
                  string,
              ][])
            : []),
    ];
    const sourceLabel = multi
        ? `${sources.length} sources`
        : `#${singleSource?.id ?? "?"}`;
    const removedLabel = removedSources.map((id) => `#${id}`).join(", ");
    const keptLabel = keptSources
        .map(
            (source) =>
                `#${source.id} keeps ${source.frames} frame${
                    source.frames === 1 ? "" : "s"
                } outside the selection`,
        )
        .join(" · ");

    return (
        <Dialog
            title={
                multi
                    ? `Merge ${sources.length} ${vocab.units} into #${target.id}`
                    : `Merge ${vocab.unit} #${singleSource?.id} into #${target.id}`
            }
            onClose={onCancel}
            wide
            footer={
                <>
                    <Button variant="ghost" onClick={onCancel}>
                        Cancel
                    </Button>
                    <Button
                        variant="primary"
                        disabled={!canMerge}
                        onClick={() => onConfirm(plan)}
                    >
                        {multi
                            ? `Merge ${sources.length} into #${target.id}`
                            : `Merge #${singleSource?.id} into #${target.id}`}
                    </Button>
                </>
            }
        >
            <div className={styles.body}>
                <div className={styles.identity}>
                    <span className={styles.rowLabel}>Keeping</span>
                    <select
                        className={`input ${styles.select}`}
                        value={target.id}
                        onChange={(event) =>
                            onTargetChange(Number(event.target.value))
                        }
                        aria-label="Tracklet that keeps its identity"
                    >
                        {allTracklets.map((tracklet) => (
                            <option key={tracklet.id} value={tracklet.id}>
                                {tracklet.label} #{tracklet.id} ·{" "}
                                {tracklet.maskFrames.count} frames
                            </option>
                        ))}
                    </select>
                </div>

                <div className={styles.sources}>
                    <span className={styles.rowLabel}>Absorbing</span>
                    {sources.map((source) => (
                        <span key={source.id} className={styles.sourceChip}>
                            <span
                                className={styles.swatch}
                                style={{ background: source.color }}
                            />
                            {source.label} #{source.id} ·{" "}
                            {source.maskFrames.count}f
                        </span>
                    ))}
                </div>

                <div className={styles.row}>
                    <span className={styles.rowLabel}>Scope</span>
                    {multi ? (
                        <span className={styles.note}>
                            Merging {sources.length} tracklets uses their whole
                            histories. Use a two-tracklet selection for a frame
                            or range merge.
                        </span>
                    ) : (
                        <>
                            <div
                                className="segmented"
                                role="radiogroup"
                                aria-label="Merge scope"
                            >
                                {(
                                    [
                                        ["all", "Whole tracklet"],
                                        ["frame", "This frame"],
                                        ["range", "Range"],
                                    ] as const
                                ).map(([kind, label]) => (
                                    <button
                                        key={kind}
                                        type="button"
                                        role="radio"
                                        aria-checked={scopeKind === kind}
                                        className={`segment ${scopeKind === kind ? "segmentActive" : ""}`}
                                        onClick={() => setScopeKind(kind)}
                                    >
                                        {label}
                                    </button>
                                ))}
                            </div>
                            {scopeKind === "frame" && (
                                <span className={styles.note}>
                                    Only frame {currentFrame + 1}: both
                                    tracklets remain unless #{singleSource?.id}{" "}
                                    has no other masks.
                                </span>
                            )}
                        </>
                    )}
                    {!multi && scopeKind === "range" && (
                        <div className={styles.range}>
                            Frames
                            <input
                                className={`input ${styles.rangeInput}`}
                                type="number"
                                min={1}
                                max={clip.frameCount}
                                value={rangeFirst}
                                onChange={(event) =>
                                    setRangeFirst(Number(event.target.value))
                                }
                                aria-label="First frame of the merge range"
                            />
                            to
                            <input
                                className={`input ${styles.rangeInput}`}
                                type="number"
                                min={1}
                                max={clip.frameCount}
                                value={rangeLast}
                                onChange={(event) =>
                                    setRangeLast(Number(event.target.value))
                                }
                                aria-label="Last frame of the merge range"
                            />
                            of {clip.frameCount}
                        </div>
                    )}
                    {rangeInvalid && (
                        <span className={styles.error}>
                            Enter a valid range inside 1–{clip.frameCount}.
                        </span>
                    )}
                </div>

                {plan.stats.conflicts > 0 && (
                    <div className={styles.row}>
                        <span className={styles.rowLabel}>
                            {plan.stats.conflicts} overlapping frame
                            {plan.stats.conflicts === 1 ? "" : "s"}
                        </span>
                        <div
                            className="segmented"
                            role="radiogroup"
                            aria-label="Overlap policy"
                        >
                            {policyOptions.map(([value, label]) => (
                                <button
                                    key={value}
                                    type="button"
                                    role="radio"
                                    aria-checked={policy === value}
                                    className={`segment ${policy === value ? "segmentActive" : ""}`}
                                    onClick={() => setPolicy(value)}
                                >
                                    {label}
                                </button>
                            ))}
                        </div>
                        {policy === null && (
                            <span className={styles.warn}>
                                Choose how to combine the overlapping masks
                                before merging.
                            </span>
                        )}
                    </div>
                )}

                {labelsDiffer && (
                    <label className={styles.row}>
                        <span className={styles.rowLabel}>Label</span>
                        <select
                            className={`input ${styles.select}`}
                            value={labelChoice === null ? "" : labelChoice}
                            onChange={(event) =>
                                setLabelChoice(
                                    event.target.value === ""
                                        ? null
                                        : Number(event.target.value),
                                )
                            }
                        >
                            <option value="">Unlabelled</option>
                            {clip.labels.map((label) => (
                                <option key={label.id} value={label.id}>
                                    {label.name}
                                </option>
                            ))}
                        </select>
                        <span className={styles.note}>
                            The selected tracklets carry different labels. The
                            merged object keeps the one chosen here.
                        </span>
                    </label>
                )}

                <div className={styles.stats}>
                    <div className={styles.stat}>
                        <span className={styles.statLabel}>
                            Only #{target.id}
                        </span>
                        <span className={styles.statValue}>
                            {plan.stats.targetOnly} frames
                        </span>
                    </div>
                    <div className={styles.stat}>
                        <span className={styles.statLabel}>
                            Filled from {sourceLabel}
                        </span>
                        <span className={styles.statValue}>
                            {plan.stats.sourceOnly} frames
                        </span>
                    </div>
                    <div className={styles.stat}>
                        <span className={styles.statLabel}>Overlapping</span>
                        <span className={styles.statValue}>
                            {plan.stats.conflicts} frames
                        </span>
                    </div>
                    <div className={styles.stat}>
                        <span className={styles.statLabel}>Result</span>
                        <span className={styles.statValue}>
                            {resultFrames} frames ·{" "}
                            {plan.stats.resultArea.toLocaleString()} px
                        </span>
                    </div>
                    <div className={styles.stat}>
                        <span className={styles.statLabel}>Human</span>
                        <span className={styles.statValue}>
                            {plan.stats.human} frames
                        </span>
                    </div>
                    <div className={styles.stat}>
                        <span className={styles.statLabel}>Propagated</span>
                        <span className={styles.statValue}>
                            {plan.stats.propagated} frames
                        </span>
                    </div>
                </div>

                {plan.stats.downgraded > 0 && (
                    <p className={styles.warn}>
                        {plan.stats.downgraded} hand-verified frame
                        {plan.stats.downgraded === 1 ? "" : "s"} will become
                        propagated because the merge combines them with
                        unverified masks.
                    </p>
                )}

                <p className={styles.note}>
                    {removedLabel
                        ? `${removedLabel} will be removed from the project.`
                        : keptLabel}
                    {keptLabel && removedLabel ? ` ${keptLabel}.` : ""}
                    {keptSources.length > 0 && scopeKind !== "all"
                        ? " Both tracklets keep their own masks inside the selection."
                        : ""}{" "}
                    {plan.stats.sourceOnly > 0
                        ? `${plan.stats.sourceOnly} frame${
                              plan.stats.sourceOnly === 1 ? "" : "s"
                          } only the sources cover will be copied to #${target.id}.`
                        : ""}
                </p>
            </div>
        </Dialog>
    );
}
