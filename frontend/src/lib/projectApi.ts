/**
 * Client for `POST /api/projects/create`: uploads one video and receives the
 * finished project archive back as a Blob.
 *
 * `XMLHttpRequest` is used instead of `fetch` because it exposes upload
 * progress, which matters for multi-hundred-megabyte videos.
 */

import { API_BASE } from "./apiBase";
import type { ProjectMode } from "./project";

const CREATE_ENDPOINT = `${API_BASE}/api/projects/create`;

export interface CreateProjectOptions {
	video: File;
	mode: ProjectMode;
	name?: string;
	frameStep?: number;
	jpegQuality?: number;
	maxFrames?: number;
	signal?: AbortSignal;
	onProgress?: (progress: CreateProjectProgress) => void;
}

export type CreateProjectProgress =
	| { stage: "uploading"; loaded: number; total: number }
	| { stage: "processing" }
	| { stage: "downloading"; loaded: number; total: number };

export interface CreatedProject {
	/** The archive, ready to be saved to disk and/or opened in the reviewer. */
	blob: Blob;
	fileName: string;
	name: string;
	mode: ProjectMode;
	frameCount: number | null;
	fps: number | null;
}

export class CreateProjectError extends Error {
	readonly status: number;
	constructor(message: string, status: number) {
		super(message);
		this.name = "CreateProjectError";
		this.status = status;
	}
}

function parseFileName(disposition: string | null, fallback: string): string {
	if (!disposition) return fallback;
	const utf8 = /filename\*=UTF-8''([^;]+)/i.exec(disposition);
	if (utf8) {
		try {
			return decodeURIComponent(utf8[1]);
		} catch {
			/* fall through */
		}
	}
	const plain = /filename="?([^";]+)"?/i.exec(disposition);
	return plain ? plain[1] : fallback;
}

async function errorMessageFromBlob(blob: Blob, status: number): Promise<string> {
	try {
		const text = await blob.text();
		const parsed = JSON.parse(text) as { detail?: unknown };
		if (typeof parsed.detail === "string") return parsed.detail;
		if (Array.isArray(parsed.detail)) {
			return parsed.detail
				.map((item) =>
					typeof item === "object" && item && "msg" in item
						? String((item as { msg: unknown }).msg)
						: JSON.stringify(item),
				)
				.join("; ");
		}
		if (text.trim()) return text.slice(0, 300);
	} catch {
		/* not JSON */
	}
	return `Project creation failed (${status}).`;
}

export function createProject(
	options: CreateProjectOptions,
): Promise<CreatedProject> {
	const form = new FormData();
	form.append("video", options.video, options.video.name);
	form.append("mode", options.mode);
	if (options.name?.trim()) form.append("name", options.name.trim());
	if (options.frameStep && options.frameStep > 1)
		form.append("frame_step", String(options.frameStep));
	if (options.jpegQuality)
		form.append("jpeg_quality", String(options.jpegQuality));
	if (options.maxFrames) form.append("max_frames", String(options.maxFrames));

	return new Promise<CreatedProject>((resolve, reject) => {
		const xhr = new XMLHttpRequest();
		xhr.open("POST", CREATE_ENDPOINT);
		xhr.responseType = "blob";

		const abort = () => {
			xhr.abort();
			reject(new DOMException("Project creation aborted", "AbortError"));
		};
		if (options.signal) {
			if (options.signal.aborted) {
				abort();
				return;
			}
			options.signal.addEventListener("abort", abort, { once: true });
		}

		xhr.upload.onprogress = (event) => {
			if (!options.onProgress) return;
			options.onProgress({
				stage: "uploading",
				loaded: event.loaded,
				total: event.lengthComputable ? event.total : options.video.size,
			});
		};
		xhr.upload.onload = () => options.onProgress?.({ stage: "processing" });
		xhr.onprogress = (event) => {
			if (!options.onProgress || xhr.status >= 400) return;
			options.onProgress({
				stage: "downloading",
				loaded: event.loaded,
				total: event.lengthComputable ? event.total : 0,
			});
		};

		xhr.onerror = () =>
			reject(
				new CreateProjectError(
					"Could not reach the backend. Is the FastAPI service running?",
					0,
				),
			);

		xhr.onload = async () => {
			options.signal?.removeEventListener("abort", abort);
			const blob = xhr.response as Blob;
			if (xhr.status < 200 || xhr.status >= 300) {
				reject(
					new CreateProjectError(
						await errorMessageFromBlob(blob, xhr.status),
						xhr.status,
					),
				);
				return;
			}
			const headerMode = xhr.getResponseHeader("X-Project-Mode");
			const mode: ProjectMode =
				headerMode === "instance" || headerMode === "semantic"
					? headerMode
					: options.mode;
			const name =
				xhr.getResponseHeader("X-Project-Name") ??
				options.video.name.replace(/\.[^.]+$/, "");
			const frames = Number(xhr.getResponseHeader("X-Project-Frames"));
			const fps = Number(xhr.getResponseHeader("X-Project-Fps"));
			resolve({
				blob,
				fileName: parseFileName(
					xhr.getResponseHeader("Content-Disposition"),
					`${name}.zip`,
				),
				name,
				mode,
				frameCount: Number.isFinite(frames) && frames > 0 ? frames : null,
				fps: Number.isFinite(fps) && fps > 0 ? fps : null,
			});
		};

		xhr.send(form);
	});
}

export function saveBlob(blob: Blob, fileName: string): void {
	const url = URL.createObjectURL(blob);
	const anchor = document.createElement("a");
	anchor.href = url;
	anchor.download = fileName;
	document.body.appendChild(anchor);
	anchor.click();
	anchor.remove();
	window.setTimeout(() => URL.revokeObjectURL(url), 10_000);
}
