import { useEffect, useRef, useState } from "react";
import type { KeyboardEvent as ReactKeyboardEvent } from "react";
import type { Taxonomy } from "../types";
import type { Taxonomy as RankedTaxonomy } from "../types/annotations";
import {
    getTaxonDetailBySuggestion,
    isAbort,
    suggestTaxa,
    type TaxonSuggestion,
} from "../services/TaxonomyService";
import styles from "./TaxonAutocomplete.module.css";

/** Convert a rank-indexed taxonomy from the lookup service into the flat
 * taxonomy shape used by labels. */
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

export interface TaxonAutocompleteProps {
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
export function TaxonAutocomplete({
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
