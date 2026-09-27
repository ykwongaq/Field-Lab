import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import type { SamStatus } from "../lib/samApi";
import type { ProjectMode } from "../lib/project";
import { MODE_VOCABULARY } from "../lib/project";
import styles from "./Toolbar.module.css";

export type Tool = "review" | "addMask" | "editMask";
export type DrawMethod = "sam" | "polygon" | "brush";

export type DeleteScope = "frame" | "tracklet";

interface ToolbarProps {
	mode: ProjectMode;
	tool: Tool;
	sam: SamStatus | null;
	/** The selected tracklet has a mask on the current frame. */
	modelName: string;
	canDeleteFrame: boolean;
	canEdit: boolean;
	canDeleteTracklet: boolean;
	selectedMaskCount: number;
	onToolChange: (tool: Tool) => void;
	onDelete: (scope: DeleteScope) => void;
	onRefreshStatus: () => void;
}

/**
 * Vertical function bar on the left of the video: Add mask, 
 * Delete mask, Propagate, Run model.
 */
export function Toolbar(props: ToolbarProps) {
	const vocab = MODE_VOCABULARY[props.mode];
	const [deleteOpen, setDeleteOpen] = useState(false);
	const deleteRef = useRef<HTMLDivElement>(null);

	// Close the delete menu on outside click / Escape.
	useEffect(() => {
		if (!deleteOpen) return;
		const onPointerDown = (event: MouseEvent) => {
			if (!deleteRef.current?.contains(event.target as Node))
				setDeleteOpen(false);
		};
		const onKey = (event: KeyboardEvent) => {
			if (event.key === "Escape") setDeleteOpen(false);
		};
		document.addEventListener("mousedown", onPointerDown);
		document.addEventListener("keydown", onKey);
		return () => {
			document.removeEventListener("mousedown", onPointerDown);
			document.removeEventListener("keydown", onKey);
		};
	}, [deleteOpen]);

	const model = props.modelName;
	const samNote = !props.sam
		? `Checking ${model}…`
		: props.sam.available
			? `${model} clicks, polygon or brush.`
			: `${model} is unavailable (${props.sam.error ?? "unknown reason"}); polygon and brush still work.`;
	const addHelp =
		props.mode === "semantic"
			? `Draw a mask for a class on the current frame — ${samNote} Adds to an existing class or creates a new one.`
			: `Draw a new ${vocab.unit} on the current frame — ${samNote}`;
	const editHelp = props.canEdit
		? `Change the selected ${vocab.unit}'s mask on this frame with ${model} clicks, polygon or brush (add or erase).`
		: `Select a ${vocab.unit} first (click its mask on the frame or pick it in the list).`;

	return (
		<nav className={styles.bar} aria-label="Tools">
			<ToolButton
				label="Add mask"
				shortcut="A"
				active={props.tool === "addMask"}
				title={addHelp}
				onClick={() =>
					props.onToolChange(props.tool === "addMask" ? "review" : "addMask")
				}
				icon={
					<svg viewBox="0 0 24 24" aria-hidden="true">
						<path d="M12 5v14M5 12h14" />
					</svg>
				}
			/>

			<ToolButton
				label="Edit mask"
				shortcut="E"
				active={props.tool === "editMask"}
				disabled={!props.canEdit}
				title={editHelp}
				onClick={() =>
					props.onToolChange(props.tool === "editMask" ? "review" : "editMask")
				}
				icon={
					<svg viewBox="0 0 24 24" aria-hidden="true">
						<path d="M4 20h4l10-10-4-4L4 16v4zM13 7l4 4" />
					</svg>
				}
			/>

			<div ref={deleteRef} className={styles.menuAnchor}>
				<ToolButton
					label="Delete mask"
					active={deleteOpen}
					disabled={!props.canDeleteTracklet}
					title={
						props.canDeleteTracklet
							? `Remove the selected ${vocab.unit}'s mask on this frame, or the whole ${vocab.unit}.`
							: `Select a ${vocab.unit} first.`
					}
					onClick={() => setDeleteOpen((open) => !open)}
					icon={
						<svg viewBox="0 0 24 24" aria-hidden="true">
							<path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13M10 11v6M14 11v6" />
						</svg>
					}
				/>
				{deleteOpen && (
					<div className={styles.menu} role="menu">
						<button
							type="button"
							role="menuitem"
							className={styles.menuItem}
							disabled={!props.canDeleteFrame}
							onClick={() => {
								setDeleteOpen(false);
								props.onDelete("frame");
							}}
						>
							<span>This frame only</span>
							<span className={styles.menuHint}>
								{props.canDeleteFrame
									? props.selectedMaskCount <= 1
										? `Last mask — removes the ${vocab.unit}`
										: `${props.selectedMaskCount - 1} frame(s) keep their mask`
									: "No mask on this frame"}
							</span>
						</button>
						<button
							type="button"
							role="menuitem"
							className={`${styles.menuItem} ${styles.menuDanger}`}
							onClick={() => {
								setDeleteOpen(false);
								props.onDelete("tracklet");
							}}
						>
							<span>Whole {vocab.unit}</span>
							<span className={styles.menuHint}>
								All {props.selectedMaskCount} mask(s) and its review
							</span>
						</button>
					</div>
				)}
			</div>

			<div className={styles.separator} />

			<ToolButton
				label="Propagate"
				disabled
				title="Coming next: extend the selected mask over the following frames with the SAM 2 video predictor (a bounded range, not the whole clip)."
				onClick={() => {}}
				icon={
					<svg viewBox="0 0 24 24" aria-hidden="true">
						<path d="M4 12h12M12 6l6 6-6 6" />
					</svg>
				}
			/>
			<ToolButton
				label="Run model"
				disabled
				title="Coming later: run the segmentation pipeline on the whole clip."
				onClick={() => {}}
				icon={
					<svg viewBox="0 0 24 24" aria-hidden="true">
						<path d="M7 5v14l11-7z" />
					</svg>
				}
			/>

			<div className={styles.spacer} />

			<button
				type="button"
				className={styles.status}
				onClick={props.onRefreshStatus}
				title={
					props.sam
						? props.sam.available
							? `${model} ready — ${props.sam.model} on ${props.sam.device}${
									props.sam.loaded ? "" : " (loads on first use)"
								}. Click to re-check.`
							: `${model} unavailable — ${props.sam.error ?? "unknown reason"}. Click to re-check.`
						: `Checking ${model}…`
				}
			>
				<span
					className={`${styles.dot} ${
						!props.sam
							? styles.dotPending
							: props.sam.available
								? styles.dotOk
								: styles.dotBad
					}`}
				/>
				<span className={styles.statusText}>{model}</span>
			</button>
		</nav>
	);
}

interface ToolButtonProps {
	label: string;
	icon: ReactNode;
	title: string;
	onClick: () => void;
	active?: boolean;
	disabled?: boolean;
	shortcut?: string;
}

function ToolButton(props: ToolButtonProps) {
	return (
		<button
			type="button"
			className={`${styles.tool} ${props.active ? styles.toolActive : ""}`}
			disabled={props.disabled}
			aria-pressed={props.active}
			title={props.shortcut ? `${props.title} (${props.shortcut})` : props.title}
			onClick={props.onClick}
		>
			<span className={styles.icon}>{props.icon}</span>
			<span className={styles.label}>{props.label}</span>
		</button>
	);
}
