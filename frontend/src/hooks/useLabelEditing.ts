import {
    useCallback,
    useState,
    type Dispatch,
    type SetStateAction,
} from "react";
import { emptyTaxonomy, type Clip } from "../lib/clip";
import type { LabelStore } from "../lib/labelStore";
import { nextUnusedLabelColor } from "../lib/palette";
import type { Label, Taxonomy } from "../types";

/** Which label the editor is open on: a `null` id creates, a number edits. */
export interface LabelEditorTarget {
    labelId: number | null;
}

/** The values the label editor writes back on save. */
export interface LabelPatch {
    name: string;
    taxonomy: Taxonomy;
    color: string;
}

export interface UseLabelEditingParams {
    clip: Clip;
    setClip: Dispatch<SetStateAction<Clip>>;
    store: LabelStore;
    /** Bump the render so the mutable store's new value is read again. */
    refresh: () => void;
    /** Move the selection, for "jump to the first object of this label". */
    selectTracklet: (id: number) => void;
    /** Report a label deletion in the workspace's notice bar. */
    onNotice: (text: string) => void;
}

export interface LabelEditing {
    /** Instance mode: the label a newly drawn object is assigned to. */
    newLabelId: number | null;
    setNewLabelId: Dispatch<SetStateAction<number | null>>;
    labelEditor: LabelEditorTarget | null;
    setLabelEditor: Dispatch<SetStateAction<LabelEditorTarget | null>>;
    /** The label whose deletion is awaiting confirmation. */
    pendingLabelDelete: number | null;
    setPendingLabelDelete: Dispatch<SetStateAction<number | null>>;
    /** The label the editor is open on, or `null` when creating. */
    editorLabel: Label | null;
    editorTaxonomy: Taxonomy;
    editorColor: string;
    /** The label a delete confirmation is about, and how many it would free. */
    pendingLabel: Label | null;
    pendingLabelCount: number;
    assignTrackletLabel: (trackletId: number, labelId: number | null) => void;
    createLabelForTracklet: (trackletId: number) => void;
    openLabelEditor: (labelId: number | null) => void;
    saveLabel: (patch: LabelPatch) => void;
    deleteLabel: (labelId: number) => void;
    selectLabel: (labelId: number) => void;
}

/**
 * Label table editing: the editor dialog, the picker's "＋" target, the pending
 * delete, and the per-object assignment.
 *
 * Label ids are positional, so deleting one renumbers the rest — the store is
 * keyed by id and has to be shifted alongside the clip, and the picker's target
 * id has to follow. That bookkeeping is the reason these handlers belong together
 * rather than scattered through the component.
 */
export function useLabelEditing({
    clip,
    setClip,
    store,
    refresh,
    selectTracklet,
    onNotice,
}: UseLabelEditingParams): LabelEditing {
    const [newLabelId, setNewLabelId] = useState<number | null>(null);
    const [labelEditor, setLabelEditor] = useState<LabelEditorTarget | null>(
        null,
    );
    const [pendingLabelDelete, setPendingLabelDelete] = useState<number | null>(
        null,
    );

    /** Assign a label to one object, or move it to unlabelled. */
    const assignTrackletLabel = useCallback(
        (trackletId: number, labelId: number | null) => {
            setClip(clip.setLabel(trackletId, labelId));
            refresh();
        },
        [clip, setClip, refresh],
    );

    /** A name no other label is already using. */
    const uniqueLabelName = useCallback(
        (base: string) => {
            let name = base;
            let suffix = 2;
            while (
                clip.labels.some(
                    (item) => item.name.toLowerCase() === name.toLowerCase(),
                )
            ) {
                name = `${base} ${suffix}`;
                suffix += 1;
            }
            return name;
        },
        [clip.labels],
    );

    /** The picker's "＋": create a label, assign it, then open its editor. */
    const createLabelForTracklet = useCallback(
        (trackletId: number) => {
            const { clip: assigned, label } = clip.assignLabel(
                trackletId,
                uniqueLabelName("new label"),
            );
            // A fresh label must not inherit a saved edit left behind by an
            // earlier label that happened to hold the same id.
            store.remove(label.id);
            setClip(assigned);
            refresh();
            setLabelEditor({ labelId: label.id });
        },
        [clip, store, setClip, uniqueLabelName, refresh],
    );

    const openLabelEditor = useCallback(
        (labelId: number | null) => setLabelEditor({ labelId }),
        [],
    );

    /** Save the label editor: add a new label, or update the one being edited. */
    const saveLabel = useCallback(
        (patch: LabelPatch) => {
            if (!labelEditor) return;
            if (labelEditor.labelId === null) {
                const { clip: added, label } = clip.addLabel(patch.name);
                // Clear any saved edit left by a previously deleted label that
                // held this id, then write the values just entered.
                store.remove(label.id);
                setClip(
                    added.updateLabel(label.id, {
                        taxonomy: patch.taxonomy,
                        color: patch.color,
                    }),
                );
                store.set(label.id, patch.taxonomy);
                store.setColor(label.id, patch.color);
            } else {
                const labelId = labelEditor.labelId;
                setClip(
                    clip.updateLabel(labelId, {
                        name: patch.name,
                        taxonomy: patch.taxonomy,
                        color: patch.color,
                    }),
                );
                store.set(labelId, patch.taxonomy);
                store.setColor(labelId, patch.color);
            }
            setLabelEditor(null);
            refresh();
        },
        [clip, labelEditor, store, setClip, refresh],
    );

    const deleteLabel = useCallback(
        (labelId: number) => {
            const affected = clip.tracklets.filter(
                (tracklet) => tracklet.labelId === labelId,
            ).length;
            const next = clip.deleteLabel(labelId);
            if (next === clip) {
                setPendingLabelDelete(null);
                return;
            }
            // The store is keyed by label id, and ids are positional, so it has
            // to renumber alongside the clip.
            store.deleteLabel(labelId);
            setNewLabelId((current) => {
                if (current === null) return null;
                if (current === labelId) return null;
                return current > labelId ? current - 1 : current;
            });
            setClip(next);
            setPendingLabelDelete(null);
            refresh();
            onNotice(
                affected === 0
                    ? "Label deleted."
                    : `Label deleted — ${affected} ${
                          affected === 1 ? "object is" : "objects are"
                      } now unlabelled.`,
            );
        },
        [clip, store, setClip, refresh, onNotice],
    );

    /** Select the first object that uses a label. */
    const selectLabel = useCallback(
        (labelId: number) => {
            const first = clip.tracklets.find(
                (tracklet) => tracklet.labelId === labelId,
            );
            if (first) selectTracklet(first.id);
        },
        [clip.tracklets, selectTracklet],
    );

    const editorLabel =
        labelEditor?.labelId != null
            ? clip.labelById(labelEditor.labelId)
            : null;
    const editorTaxonomy = editorLabel
        ? store.get(editorLabel)
        : emptyTaxonomy();
    /** A new label starts on the next colour no other label is using. */
    const editorColor = editorLabel
        ? store.colorOf(editorLabel)
        : nextUnusedLabelColor(clip.labels.map((label) => label.color));

    const pendingLabel =
        pendingLabelDelete === null ? null : clip.labelById(pendingLabelDelete);
    const pendingLabelCount =
        pendingLabelDelete === null
            ? 0
            : clip.tracklets.filter(
                  (tracklet) => tracklet.labelId === pendingLabelDelete,
              ).length;

    return {
        newLabelId,
        setNewLabelId,
        labelEditor,
        setLabelEditor,
        pendingLabelDelete,
        setPendingLabelDelete,
        editorLabel,
        editorTaxonomy,
        editorColor,
        pendingLabel,
        pendingLabelCount,
        assignTrackletLabel,
        createLabelForTracklet,
        openLabelEditor,
        saveLabel,
        deleteLabel,
        selectLabel,
    };
}
