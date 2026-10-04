"""Who a request belongs to.

There is no authentication here, and this module is not it. What it adds is
*partitioning*: every session and every propagation job records the client that
created it, and nothing can be read, polled or cancelled without presenting that
same id. That is what stops one reviewer's frames, masks and runs from turning up
in another reviewer's UI — which is a bug the moment a second person uses the same
server, and why the id is required rather than optional.

Be precise about what that buys. The id is self-declared, so anyone who learns
someone else's id can act as them: treat it as a tenant key, not a credential. It
is fine behind a trusted network or on a shared lab machine, and it is not fine
exposed to people you do not trust. If that ever changes, replace the header's
*value* with a verified identity from real auth; nothing else in the design has
to move.

A session id is deliberately not usable as this id. It is a resource name that
already travels in a URL path (`/api/sessions/{id}/frames/{n}`), so it lands in
access logs, browser history and `Referer` headers — and a credential must never
be a URL segment.
"""

from __future__ import annotations

import re
import uuid
from typing import Optional

from src.core.errors import InvalidRequest

#: The header every session, prompt and propagation request must carry.
CLIENT_HEADER = "X-Vsr-Client"

#: 32 hex characters (a UUID4 without its dashes). The length is what makes an id
#: unguessable enough to be worth having, and the charset keeps it safe to compare
#: and to store in the session's JSON.
CLIENT_ID_PATTERN = re.compile(r"\A[0-9a-fA-F]{32}\Z")


def validate_client_id(raw: Optional[str]) -> str:
    """Return the normalised client id, or explain what the request is missing."""
    client_id = (raw or "").strip().lower()
    if not CLIENT_ID_PATTERN.match(client_id):
        raise InvalidRequest(
            f"This endpoint needs a `{CLIENT_HEADER}` header holding a "
            "32-character hex client id, so the backend can keep one reviewer's "
            "sessions, frames and runs out of another reviewer's reach. Generate "
            "one per browser (see `frontend/src/lib/clientId.ts`) and send it with "
            "every request."
        )
    return client_id


def new_client_id() -> str:
    """A fresh client id. Used by tests, and by anything that needs to mint one."""
    return uuid.uuid4().hex
