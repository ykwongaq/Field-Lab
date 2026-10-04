/** Muted, colour-blind-friendly palette used to colour tracklets. */
const PALETTE = [
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#dbdb8d",
    "#9edae5",
    "#393b79",
    "#637939",
    "#8c6d31",
    "#843c39",
    "#7b4173",
];

export function colorForIndex(index: number): string {
    return PALETTE[
        ((index % PALETTE.length) + PALETTE.length) % PALETTE.length
    ];
}

/**
 * Label colours: `PALETTE` without the reds.
 *
 * Red is reserved for "no label", so a label must never be assigned a colour
 * that could be mistaken for it. This is a separate list rather than a change to
 * `PALETTE`, so per-object colours keep their indices.
 */
export const LABEL_PALETTE = PALETTE.filter(
    (color) =>
        color !== "#d62728" && color !== "#ff9896" && color !== "#843c39",
);

/**
 * Colour for label number `index` (0-based), cycling when the palette runs out.
 *
 * Labels own a colour, so every mask of one class is drawn in it. The number on
 * each colour block is what disambiguates labels that cycled back onto the same
 * colour.
 */
export function labelColorFor(index: number): string {
    return LABEL_PALETTE[
        ((index % LABEL_PALETTE.length) + LABEL_PALETTE.length) %
            LABEL_PALETTE.length
    ];
}

/**
 * The first palette colour no existing label is using, so a new label's default
 * keeps changing as labels are added. Once every colour is taken it cycles.
 */
export function nextUnusedLabelColor(used: Iterable<string>): string {
    const taken = new Set(used);
    for (const color of LABEL_PALETTE) {
        if (!taken.has(color)) return color;
    }
    return labelColorFor(taken.size);
}

/**
 * The colour reserved for the selected object's mask.
 *
 * It is deliberately absent from `PALETTE`, so no tracklet can ever be drawn in
 * it. On the canvas it therefore means exactly one thing: "this is the object
 * you are working on". Every other mask keeps its own colour, which is how you
 * spot two animals being confused.
 */
export const SELECTED_COLOR = "#1f77b4";

/**
 * The colour reserved for objects that carry no label.
 *
 * Red reads as "needs attention", which is exactly the state of an unlabelled
 * mask. Like `SELECTED_COLOR` it is deliberately absent from `PALETTE`, so no
 * labelled tracklet can ever be drawn in it.
 */
export const UNLABELLED_COLOR = "#ff0000";
