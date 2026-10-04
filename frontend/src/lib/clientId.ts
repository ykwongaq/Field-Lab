/**
 * The client id every backend request carries.
 *
 * The backend keeps sessions, frames and propagation jobs apart by this id (see
 * `backend/src/core/identity.py`), so it has to be stable per *browser*: stored
 * in `localStorage` so a reload — or coming back to a long propagation — still
 * finds its own work, and shared by every tab so two tabs are one reviewer
 * rather than two.
 *
 * It is a partition key, not a credential. The backend does not authenticate it,
 * so anyone who learns this value can act as this browser. That is fine against a
 * trusted server and not fine on the open internet; if the backend ever grows
 * real auth, the verified identity takes this header's place and nothing else
 * here changes.
 */

const STORAGE_KEY = "vsr.clientId";

/** 32 hex characters — the format the backend validates. */
const ID_PATTERN = /^[0-9a-f]{32}$/;

/**
 * `crypto.getRandomValues` rather than `crypto.randomUUID`, which only exists in
 * a secure context: this way an http:// origin on a lab machine still works.
 */
function randomId(): string {
    const bytes = new Uint8Array(16);
    crypto.getRandomValues(bytes);
    return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join(
        "",
    );
}

/**
 * Module-level so a storage failure still yields one id for the page's lifetime,
 * instead of a fresh id per request — which would look to the backend like a new
 * reviewer every time and hide the session that was just created.
 */
let cached: string | null = null;

/** This browser's client id, generating and persisting one on first use. */
export function clientId(): string {
    if (cached) return cached;

    try {
        const stored = localStorage.getItem(STORAGE_KEY);
        if (stored && ID_PATTERN.test(stored)) {
            cached = stored;
            return cached;
        }
    } catch {
        /* storage is blocked (private mode, sandboxed frame): mint a fresh id */
    }

    cached = randomId();
    try {
        localStorage.setItem(STORAGE_KEY, cached);
    } catch {
        /* the id still works for as long as this page lives */
    }
    return cached;
}
