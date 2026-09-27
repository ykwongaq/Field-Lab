import { useMemo, useState } from "react";
import type { Clip } from "../lib/clip";
import type { TrackletReview } from "../types";
import { MODE_VOCABULARY } from "../lib/project";
import styles from "./TrackletList.module.css";

type Status = "done" | "partial" | "todo";
type Filter = "all" | "todo" | "partial" | "done";

const STATUS_LABEL: Record<Status, string> = {
	done: "Verified",
	partial: "In progress",
	todo: "Pending",
};

interface TrackletListProps {
	clip: Clip;
	selectedId: number | null;
	reviews: Record<number, TrackletReview>;
	onSelect: (id: number) => void;
}

function statusOf(review: TrackletReview | undefined): Status {
	if (!review) return "todo";
	const hasLabel = review.labelConfirmed;
	const hasMask = review.maskVerdict !== null;
	if (hasLabel && hasMask) return "done";
	if (hasLabel || hasMask) return "partial";
	return "todo";
}

export function TrackletList({
	clip,
	selectedId,
	reviews,
	onSelect,
}: TrackletListProps) {
	const [query, setQuery] = useState("");
	const [filter, setFilter] = useState<Filter>("all");
	const vocab = MODE_VOCABULARY[clip.mode];

	const items = useMemo(() => {
		const needle = query.trim().toLowerCase();
		return clip.tracklets.filter((tracklet) => {
			if (filter !== "all" && statusOf(reviews[tracklet.id]) !== filter)
				return false;
			if (!needle) return true;
			return (
				tracklet.label.toLowerCase().includes(needle) ||
				String(tracklet.id).includes(needle) ||
				String(tracklet.objectId).includes(needle)
			);
		});
	}, [clip.tracklets, query, filter, reviews]);

	return (
		<div className={styles.list}>
			<div className={styles.listHeader}>
				<input
					type="search"
					className={styles.search}
					placeholder="Filter by label or id…"
					value={query}
					onChange={(event) => setQuery(event.target.value)}
				/>
				<div className={styles.filters}>
					{(["all", "todo", "partial", "done"] as const).map((option) => (
						<button
							key={option}
							type="button"
							className={`${styles.filterBtn} ${filter === option ? styles.filterActive : ""}`}
							onClick={() => setFilter(option)}
						>
							{option === "all" ? "All" : STATUS_LABEL[option]}
						</button>
					))}
				</div>
			</div>

			<div className={styles.listScroll}>
				{clip.tracklets.length === 0 ? (
					<p className={styles.empty}>
						This project has no annotations yet. Run the segmentation
						pipeline on the embedded video, write the results into{" "}
						<code>annotations/</code>, and re-open the archive.
					</p>
				) : (
					items.length === 0 && (
						<p className={styles.empty}>No {vocab.units} match.</p>
					)
				)}
				{items.map((tracklet) => {
					const status = statusOf(reviews[tracklet.id]);
					return (
						<button
							key={tracklet.id}
							type="button"
							className={`${styles.item} ${tracklet.id === selectedId ? styles.selected : ""}`}
							onClick={() => onSelect(tracklet.id)}
						>
							<span
								className={styles.swatch}
								style={{ background: tracklet.color }}
							/>
							<span className={styles.itemBody}>
								<span className={styles.itemLabel}>
									{tracklet.label}
									{tracklet.origin === "created" && (
										<span className={styles.newTag} title="Added in this session with the Add mask tool; not in the archive yet">
											new
										</span>
									)}
								</span>
								<span className={styles.itemMeta}>
									#{tracklet.id}
									{vocab.hasObjectIdentity && <> · obj {tracklet.objectId}</>}{" "}
									· {tracklet.maskFrames.count} frames
								</span>
							</span>
							<span
								className={`${styles.status} ${
									status === "done"
										? styles.statusDone
										: status === "partial"
											? styles.statusPartial
											: styles.statusTodo
								}`}
							>
								{STATUS_LABEL[status]}
							</span>
						</button>
					);
				})}
			</div>
		</div>
	);
}
