import { clientId } from "./clientId";

export const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(
    /\/+$/,
    "",
);

/** The header the backend reads the client id from. */
const CLIENT_HEADER = "X-Vsr-Client";

/**
 * `fetch` for this backend, carrying the client id.
 *
 * Every backend call goes through here rather than through `fetch` directly, so
 * a new call site cannot forget the header: the endpoints that touch a session,
 * a prompt or a job all require it and answer 422 without it (see
 * `backend/src/core/identity.py`). Requests to other hosts use plain `fetch`.
 */
export function apiFetch(
    input: string,
    init: RequestInit = {},
): Promise<Response> {
    const headers = new Headers(init.headers);
    headers.set(CLIENT_HEADER, clientId());
    return fetch(input, { ...init, headers });
}
