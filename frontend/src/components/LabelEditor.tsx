import { useState } from "react";
import type { Label, Taxonomy, TaxonomyKey } from "../types";
import { LABEL_PALETTE } from "../lib/palette";
import { Button, Dialog } from "../ui";
import { TaxonAutocomplete } from "./TaxonAutocomplete";
import styles from "./LabelEditor.module.css";

const TAXONOMY_FIELDS: {
    key: TaxonomyKey;
    label: string;
    placeholder: string;
}[] = [
    { key: "kingdom", label: "Kingdom", placeholder: "Animalia" },
    { key: "phylum", label: "Phylum", placeholder: "Chordata" },
    { key: "class", label: "Class", placeholder: "Mammalia" },
    { key: "order", label: "Order", placeholder: "Primates" },
    { key: "family", label: "Family", placeholder: "Atelidae" },
    { key: "genus", label: "Genus", placeholder: "Ateles" },
    { key: "species", label: "Species", placeholder: "Ateles geoffroyi" },
];

export interface LabelEditorProps {
    /** The label being edited, or `null` when creating a new one. */
    label: Label | null;
    /** The taxonomy in force: the saved edit, else the label's own. */
    taxonomy: Taxonomy;
    /** The colour in force: the saved edit, else the label's own. */
    color: string;
    onSave: (patch: {
        name: string;
        taxonomy: Taxonomy;
        color: string;
    }) => void;
    onClose: () => void;
}

/**
 * Create or edit a label: its name, its colour, and the taxonomy shared by
 * every object assigned to it.
 */
export function LabelEditor({
    label,
    taxonomy,
    color: initialColor,
    onSave,
    onClose,
}: LabelEditorProps) {
    // The label's name is the common name it is saved and exported under. The
    // field starts on the taxonomy's own common name; a label with no taxonomy
    // at all keeps its existing name, while a rank-only label starts empty so
    // that editing a rank can never turn the old value into a stale name.
    const hasRank = (
        [
            "kingdom",
            "phylum",
            "class",
            "order",
            "family",
            "genus",
            "species",
        ] as const
    ).some((rank) => taxonomy[rank].trim() !== "");
    const [name, setName] = useState(
        taxonomy.commonName.trim() || (hasRank ? "" : label?.name || ""),
    );
    const [color, setColor] = useState(initialColor);
    const [draft, setDraft] = useState<Taxonomy>(taxonomy);

    // A colour that is not one of the palette's own is the custom choice.
    const customColor = !LABEL_PALETTE.includes(color);

    const setField = (key: TaxonomyKey, value: string) =>
        setDraft((current) => ({ ...current, [key]: value }));

    return (
        <Dialog
            title={label ? `Edit label ${label.id}` : "New label"}
            onClose={onClose}
            wide
            footer={
                <>
                    <Button variant="ghost" onClick={onClose}>
                        Cancel
                    </Button>
                    <Button
                        variant="primary"
                        onClick={() =>
                            onSave({
                                name: name.trim() || "unlabelled object",
                                // The name field is the common name: write it
                                // onto the taxonomy too, since the export names
                                // a category from `common_name` (else the
                                // deepest rank) and ignores the label name.
                                taxonomy: {
                                    ...draft,
                                    commonName: name.trim(),
                                },
                                color,
                            })
                        }
                    >
                        {label ? "Save" : "Create"}
                    </Button>
                </>
            }
        >
            <div className={styles.body}>
                <label className={styles.row}>
                    <span className={styles.rowLabel}>Name</span>
                    <input
                        className="input"
                        value={name}
                        data-autofocus
                        placeholder="Common name, e.g. Spider Monkey"
                        onChange={(event) => setName(event.target.value)}
                        aria-label="Label name"
                    />
                </label>

                <div className={styles.row}>
                    <span className={styles.rowLabel}>Colour</span>
                    <div className={styles.swatches}>
                        <div
                            className={styles.paletteGroup}
                            role="radiogroup"
                            aria-label="Label colour"
                        >
                            {LABEL_PALETTE.map((swatch) => (
                                <button
                                    key={swatch}
                                    type="button"
                                    role="radio"
                                    aria-checked={swatch === color}
                                    aria-label={`Colour ${swatch}`}
                                    className={`${styles.swatch} ${
                                        swatch === color
                                            ? styles.swatchActive
                                            : ""
                                    }`}
                                    style={{ background: swatch }}
                                    onClick={() => setColor(swatch)}
                                />
                            ))}
                        </div>
                        {/* A free colour, past the palette: the native picker
                            keeps it to one control. */}
                        <label
                            className={`${styles.custom} ${
                                customColor ? styles.swatchActive : ""
                            }`}
                            title="Custom colour"
                        >
                            <input
                                type="color"
                                className={styles.colorInput}
                                value={color}
                                onChange={(event) =>
                                    setColor(event.target.value)
                                }
                                aria-label="Custom colour"
                            />
                            <span
                                className={styles.customPlus}
                                aria-hidden="true"
                            >
                                ＋
                            </span>
                        </label>
                    </div>
                </div>

                <div className={styles.sep} />

                <div className={styles.rows}>
                    {TAXONOMY_FIELDS.map(
                        ({ key, label: fieldLabel, placeholder }) => (
                            <div key={key} className={styles.row}>
                                <span className={styles.rowLabel}>
                                    {fieldLabel}
                                </span>
                                <TaxonAutocomplete
                                    value={draft[key]}
                                    label={fieldLabel}
                                    rank={key.toUpperCase()}
                                    placeholder={placeholder}
                                    commonName={name}
                                    onChange={(value) => setField(key, value)}
                                    onApply={setDraft}
                                />
                            </div>
                        ),
                    )}
                </div>
            </div>
        </Dialog>
    );
}
