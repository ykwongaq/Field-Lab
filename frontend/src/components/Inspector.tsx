import { useEffect, useRef, useState } from "react";
import type { KeyboardEvent as ReactKeyboardEvent } from "react";
import type { Taxonomy, TaxonomyKey, Tracklet } from "../types";
import type { Taxonomy as RankedTaxonomy } from "../types/annotations";
import { MODE_VOCABULARY, type ProjectMode } from "../lib/project";
import {
    getTaxonDetailBySuggestion,
    isAbort,
    suggestTaxa,
    type TaxonSuggestion,
} from "../services/TaxonomyService";
import { EmptyState } from "../ui";
import styles from "./Inspector.module.css";

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
    {
        key: "commonName",
        label: "Common name",
        placeholder: "Geoffroy's Spider Monkey",
    },
];

interface InspectorProps {
    mode: ProjectMode;
    tracklet: Tracklet | null;
    /** The taxonomy in force: the workspace edit, else the archive's own. */
    taxonomy: Taxonomy | null;
    onTaxonomyField: (key: TaxonomyKey, value: string) => void;
    onApplyTaxonomy: (taxonomy: Taxonomy) => void;
}

/**
 * The inspector: the identity and taxonomy of the selected object.
 *
 * Picking a suggestion in any rank fetches the taxon's full GBIF hierarchy and
 * fills that rank plus every ancestor above it, which is the whole point — the
 * species is what a curator knows, the path above it is bookkeeping.
 */
export function Inspector(props: InspectorProps) {
    const { tracklet, taxonomy } = props;
    const vocab = MODE_VOCABULARY[props.mode];
    const unitTitle = vocab.unit.charAt(0).toUpperCase() + vocab.unit.slice(1);

    if (!tracklet || !taxonomy) {
        return (
            <section className={styles.panel}>
                <header className={styles.head}>
                    <span className="sectionLabel">Inspector</span>
                </header>
                <EmptyState icon="target" title={`No ${vocab.unit} selected`}>
                    Pick one from the list, or click its mask on the frame.
                </EmptyState>
            </section>
        );
    }

    const confirmed =
        taxonomy.species.trim() !== "" || taxonomy.commonName.trim() !== "";

    return (
        <section className={styles.panel} aria-label={`${unitTitle} details`}>
            <header className={styles.head}>
                <span
                    className={styles.swatch}
                    style={{ background: tracklet.color }}
                />
                <div className={styles.headText}>
                    <h2 className={styles.title}>
                        {tracklet.label}
                        <span className={styles.id}>#{tracklet.id}</span>
                    </h2>
                    <p className={styles.meta}>
                        {vocab.hasObjectIdentity
                            ? `object ${tracklet.objectId} · `
                            : ""}
                        frames {tracklet.maskFrames.first + 1}–
                        {tracklet.maskFrames.last + 1} ·{" "}
                        {tracklet.maskFrames.count}{" "}
                        {tracklet.maskFrames.count === 1 ? "mask" : "masks"}
                    </p>
                </div>
            </header>

            <div className={`scrollArea ${styles.body}`}>
                <div className={styles.sectionHead}>
                    <span className="sectionLabel">Taxonomy</span>
                    {confirmed && <span className="chip chipGood">named</span>}
                </div>

                <div className={styles.rows}>
                    {TAXONOMY_FIELDS.map(({ key, label, placeholder }) =>
                        key === "commonName" ? (
                            <label key={key} className={styles.row}>
                                <span className={styles.rowLabel}>{label}</span>
                                <input
                                    className="input"
                                    value={taxonomy[key]}
                                    placeholder={placeholder}
                                    onChange={(event) =>
                                        props.onTaxonomyField(
                                            key,
                                            event.target.value,
                                        )
                                    }
                                />
                            </label>
                        ) : (
                            <div key={key} className={styles.row}>
                                <span className={styles.rowLabel}>{label}</span>
                                <TaxonAutocomplete
                                    value={taxonomy[key]}
                                    label={label}
                                    rank={key.toUpperCase()}
                                    placeholder={placeholder}
                                    commonName={taxonomy.commonName}
                                    onChange={(value) =>
                                        props.onTaxonomyField(key, value)
                                    }
                                    onApply={props.onApplyTaxonomy}
                                />
                            </div>
                        ),
                    )}
                </div>
            </div>
        </section>
    );
}

/** Convert a rank-indexed taxonomy from the lookup service into the flat
 * taxonomy shape used by tracklets. */
function rankedToFlatTaxonomy(
    ranked: RankedTaxonomy,
    commonName: string,
): Taxonomy {
    return {
        taxonId: ranked.sourceKey ?? null,
        kingdom: ranked.ranks.kingdom,
        phylum: ranked.ranks.phylum,
        class: ranked.ranks.class,
        order: ranked.ranks.order,
        family: ranked.ranks.family,
        genus: ranked.ranks.genus,
        species: ranked.ranks.species,
        commonName,
    };
}

interface TaxonAutocompleteProps {
    value: string;
    label: string;
    rank: string;
    placeholder: string;
    commonName: string;
    onChange: (value: string) => void;
    onApply: (taxonomy: Taxonomy) => void;
}

/**
 * Taxonomic rank input with live autocomplete. Selecting a suggestion fetches
 * the taxon's full GBIF hierarchy and fills the selected rank plus every
 * higher (ancestor) rank, leaving deeper ranks empty.
 */
function TaxonAutocomplete({
    value,
    label,
    rank,
    placeholder,
    commonName,
    onChange,
    onApply,
}: TaxonAutocompleteProps) {
    const [suggestions, setSuggestions] = useState<TaxonSuggestion[]>([]);
    const [open, setOpen] = useState(false);
    const [active, setActive] = useState(-1);
    const [applying, setApplying] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const rootRef = useRef<HTMLDivElement | null>(null);
    const suggestAbort = useRef<AbortController | null>(null);
    const suggestTimer = useRef<number | null>(null);
    const detailAbort = useRef<AbortController | null>(null);

    // Debounced suggestions are scheduled only when the user actually types,
    // so programmatic value changes never pop the dropdown open.
    const scheduleSuggest = (query: string) => {
        if (suggestTimer.current !== null) {
            window.clearTimeout(suggestTimer.current);
            suggestTimer.current = null;
        }
        suggestAbort.current?.abort();
        setError(null);

        const q = query.trim();
        if (!q) {
            setSuggestions([]);
            setOpen(false);
            setActive(-1);
            return;
        }

        const timer = window.setTimeout(() => {
            suggestAbort.current?.abort();
            const controller = new AbortController();
            suggestAbort.current = controller;
            suggestTaxa(q, controller.signal, rank)
                .then((items) => {
                    setSuggestions(items);
                    setOpen(items.length > 0);
                    setActive(items.length > 0 ? 0 : -1);
                })
                .catch((err) => {
                    if (!isAbort(err)) {
                        setError("Lookup failed");
                        setOpen(false);
                    }
                });
        }, 250);
        suggestTimer.current = timer;
    };

    const applySuggestion = async (suggestion: TaxonSuggestion) => {
        if (suggestTimer.current !== null) {
            window.clearTimeout(suggestTimer.current);
            suggestTimer.current = null;
        }
        suggestAbort.current?.abort();
        detailAbort.current?.abort();
        const controller = new AbortController();
        detailAbort.current = controller;

        setOpen(false);
        setSuggestions([]);
        setActive(-1);
        setApplying(true);
        setError(null);

        try {
            const detail = await getTaxonDetailBySuggestion(
                suggestion,
                controller.signal,
            );
            onApply(rankedToFlatTaxonomy(detail, commonName));
        } catch (err) {
            if (!isAbort(err)) setError("Could not load hierarchy");
        } finally {
            setApplying(false);
        }
    };

    const onKeyDown = (event: ReactKeyboardEvent<HTMLInputElement>) => {
        if (!open || suggestions.length === 0) return;
        if (event.key === "ArrowDown") {
            event.preventDefault();
            setActive((index) => (index + 1) % suggestions.length);
        } else if (event.key === "ArrowUp") {
            event.preventDefault();
            setActive(
                (index) =>
                    (index - 1 + suggestions.length) % suggestions.length,
            );
        } else if (event.key === "Enter") {
            if (active >= 0 && active < suggestions.length) {
                event.preventDefault();
                void applySuggestion(suggestions[active]);
            }
        } else if (event.key === "Escape") {
            event.preventDefault();
            setOpen(false);
        }
    };

    // Close the dropdown when clicking outside of the control.
    useEffect(() => {
        const onPointerDown = (event: MouseEvent) => {
            if (
                rootRef.current &&
                !rootRef.current.contains(event.target as Node)
            ) {
                setOpen(false);
            }
        };
        document.addEventListener("mousedown", onPointerDown);
        return () => document.removeEventListener("mousedown", onPointerDown);
    }, []);

    // Abort any in-flight requests and clear the debounce timer on unmount.
    useEffect(
        () => () => {
            if (suggestTimer.current !== null) {
                window.clearTimeout(suggestTimer.current);
            }
            suggestAbort.current?.abort();
            detailAbort.current?.abort();
        },
        [],
    );

    return (
        <div className={styles.autocomplete} ref={rootRef}>
            <input
                className="input"
                value={value}
                placeholder={placeholder}
                aria-label={label}
                role="combobox"
                aria-expanded={open}
                aria-autocomplete="list"
                onChange={(event) => {
                    const next = event.target.value;
                    onChange(next);
                    scheduleSuggest(next);
                }}
                onKeyDown={onKeyDown}
            />
            {applying && <span className={styles.hint}>Loading…</span>}
            {error && !open && <span className={styles.error}>{error}</span>}
            {open && suggestions.length > 0 && (
                <ul className={styles.suggestList} role="listbox">
                    {suggestions.map((suggestion, index) => (
                        <li key={suggestion.key} role="presentation">
                            <button
                                type="button"
                                role="option"
                                aria-selected={index === active}
                                className={`${styles.suggestItem} ${
                                    index === active ? styles.suggestActive : ""
                                }`}
                                onMouseDown={(event) => {
                                    event.preventDefault();
                                    void applySuggestion(suggestion);
                                }}
                                onMouseEnter={() => setActive(index)}
                            >
                                <span className={styles.suggestName}>
                                    {suggestion.scientificName}
                                </span>
                                <span className={styles.suggestRank}>
                                    {suggestion.rank.toLowerCase() || "taxon"}
                                </span>
                            </button>
                        </li>
                    ))}
                </ul>
            )}
        </div>
    );
}
