"""HTTP checks for `/api/sam3`, with a stub segmentation service.

Run from the backend root::

    python test/test_sam3_api.py

The model is replaced by a stub through FastAPI's dependency override, so this
covers what the *route* is responsible for: prompt validation, session lookup,
and the shape of the response. No GPU and no `sam3` install are needed.

Settings are forced to a throwaway temp dir before the app is imported, because
`get_settings` is cached and `main` resolves settings while being imported.
"""

import os
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image

BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_ROOT)

ROOT = tempfile.mkdtemp(prefix="vsr-sam3-api-")
TEMP_DIR = os.path.join(ROOT, "tmp")
os.environ["VSR_TEMP_DIR"] = TEMP_DIR
os.environ["VSR_LOG_DIR"] = os.path.join(ROOT, "logs")
os.environ["VSR_PROJECTS_DIR"] = os.path.join(ROOT, "projects")
os.environ["SAM3_ENABLED"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

from src.core.config import Settings, get_settings  # noqa: E402
from src.core.sessions import create_session, frame_name  # noqa: E402
from src.domain.prompts import (  # noqa: E402
    KIND_BOX,
    KIND_POINT,
    KIND_TEXT,
    InstanceMask,
    SegmentResult,
)
from src.domain.rle import decode_rle  # noqa: E402
from src.inference.registry import get_sam3_service  # noqa: E402
from src.main import app  # noqa: E402

checks = 0
HEIGHT, WIDTH = 12, 20

#: Where the stub paints its mask, so the response can be checked against it.
MASK = np.zeros((HEIGHT, WIDTH), dtype=bool)
MASK[3:7, 4:11] = True


def ok(label):
    global checks
    checks += 1
    print("  ok:", label)


class StubImageService:
    """Stands in for `Sam3ImageService`: records prompts, returns a fixed mask."""

    def __init__(self):
        self.calls = []

    def status(self):
        return {
            "available": True,
            "loaded": True,
            "loaded_models": ["image"],
            "model": "stub-sam3",
            "device": "cpu",
            "error": None,
            "cache_entries": 1,
            "point_prompts": True,
        }

    def segment(self, session, frame_index, prompt):
        self.calls.append((session.id, frame_index, prompt))
        return SegmentResult(
            mask=MASK.copy(),
            instances=[
                InstanceMask(mask=MASK.copy(), score=0.91),
                InstanceMask(mask=MASK[:6, :6].copy(), score=0.42),
            ],
            height=HEIGHT,
            width=WIDTH,
            kind=prompt.kind,
            prompt=prompt.describe(),
            encoder_ms=12.5,
            decoder_ms=3.25,
            embedding_reused=True,
        )


STUB = StubImageService()


def make_session():
    session = create_session(TEMP_DIR)
    for index in range(3):
        Image.new("RGB", (WIDTH, HEIGHT), (40, 80 + index * 20, 160)).save(
            os.path.join(session.frames_dir, frame_name(index)), format="JPEG"
        )
    return session


def check(label, condition, detail=""):
    assert condition, f"{label} {detail}"
    ok(label)


def main():
    print("sam3 api")
    client = TestClient(app)
    app.dependency_overrides[get_sam3_service] = lambda: STUB
    session = make_session()

    try:
        # -- status -------------------------------------------------------
        response = client.get("/api/sam3/status")
        check("status is 200", response.status_code == 200, response.text)
        body = response.json()
        check("status reports the model", body["model"] == "stub-sam3", body)
        check("status reports the threshold", body["threshold"] == 0.5, body)
        check("status reports point prompts", body["point_prompts"] is True, body)

        # -- a point prompt ----------------------------------------------
        STUB.calls.clear()
        response = client.post(
            "/api/sam3/segment",
            json={
                "session_id": session.id,
                "frame_index": 1,
                "prompt": {
                    "kind": "point",
                    "points": [{"x": 4, "y": 5}, {"x": 9, "y": 3, "label": 0}],
                },
            },
        )
        check("point prompt is 200", response.status_code == 200, response.text)
        body = response.json()
        decoded = decode_rle(body["rle"]["size"], body["rle"]["counts"])
        check("the mask round-trips", np.array_equal(decoded, MASK), decoded.sum())
        check("runs are sent for immediate painting", len(body["runs"]) > 0, body["runs"])
        check("the extent is reported", body["bbox"] == [4, 3, 7, 4], body["bbox"])
        check("area is reported", body["area"] == int(MASK.sum()), body["area"])
        check("timings are reported", body["decoder_ms"] == 3.2, body)
        check("the embedding cache is reported", body["embedding_reused"] is True)
        check("the prompt is echoed", "point" in body["prompt"], body["prompt"])
        check("both candidates come back", len(body["instances"]) == 2, body["instances"])
        check(
            "instance scores are ordered by the model's own verdict",
            body["instance_scores"] == [0.91, 0.42],
            body["instance_scores"],
        )
        check(
            "negative clicks reach the service",
            [p.label for p in STUB.calls[-1][2].points] == [1, 0],
            STUB.calls[-1][2].points,
        )
        check("the frame index reaches the service", STUB.calls[-1][1] == 1)

        # -- max_instances trims the alternatives -------------------------
        response = client.post(
            "/api/sam3/segment",
            json={
                "session_id": session.id,
                "frame_index": 0,
                "prompt": {"kind": "point", "points": [{"x": 5, "y": 5}]},
                "max_instances": 1,
            },
        )
        check("max_instances caps the alternatives", response.json()["instances"] == [] or
              len(response.json()["instances"]) == 1, response.json()["instances"])

        # -- a box prompt -------------------------------------------------
        response = client.post(
            "/api/sam3/segment",
            json={
                "session_id": session.id,
                "frame_index": 0,
                "prompt": {
                    "kind": "box",
                    "boxes": [{"x0": 2, "y0": 2, "x1": 12, "y1": 9}],
                },
            },
        )
        check("box prompt is 200", response.status_code == 200, response.text)
        prompt = STUB.calls[-1][2]
        check(
            "the box is parsed",
            (prompt.primary_box.x1, prompt.primary_box.y1) == (12.0, 9.0),
            prompt.primary_box,
        )
        check(
            "the box is converted to the normalised cxcywh the model wants",
            len(prompt.primary_box.to_cxcywh_normalized(WIDTH, HEIGHT)) == 4,
        )

        # -- a text prompt ------------------------------------------------
        response = client.post(
            "/api/sam3/segment",
            json={
                "session_id": session.id,
                "frame_index": 2,
                "prompt": {"kind": "text", "text": "  shark "},
            },
        )
        check("text prompt is 200", response.status_code == 200, response.text)
        check("text is trimmed", STUB.calls[-1][2].text == "shark", STUB.calls[-1][2].text)

        # -- prompt validation --------------------------------------------
        def post(prompt, frame_index=0, session_id=None):
            return client.post(
                "/api/sam3/segment",
                json={
                    "session_id": session_id or session.id,
                    "frame_index": frame_index,
                    "prompt": prompt,
                },
            )

        check(
            "a point prompt with no click is rejected",
            post({"kind": "point", "points": []}).status_code == 422,
        )
        check(
            "a click outside the frame is rejected",
            post({"kind": "point", "points": [{"x": 99, "y": 3}]}).status_code == 422,
        )
        check(
            "a negative coordinate is rejected",
            post({"kind": "point", "points": [{"x": -2, "y": 3}]}).status_code == 422,
        )
        check(
            "a box prompt with no box is rejected",
            post({"kind": "box", "boxes": []}).status_code == 422,
        )
        check(
            "a degenerate box is rejected",
            post({"kind": "box", "boxes": [{"x0": 2, "y0": 2, "x1": 2.5, "y1": 9}]}).status_code
            == 422,
        )
        check(
            "a box leaving the frame is rejected",
            post({"kind": "box", "boxes": [[0, 0, 500, 500]]}).status_code == 422,
        )
        check(
            "an empty text prompt is rejected",
            post({"kind": "text", "text": "   "}).status_code == 422,
        )
        check(
            "an unknown prompt kind is rejected",
            post({"kind": "scribble", "text": "x"}).status_code == 422,
        )
        response = post({"kind": "text", "points": [{"x": 1, "y": 1}], "text": "shark"})
        check("text mixed with clicks is rejected", response.status_code == 422, response.text)
        response = post({"kind": "point", "points": [{"x": 1, "y": 1}], "text": "shark"})
        check("clicks mixed with text is rejected", response.status_code == 422, response.text)

        # -- session handling ---------------------------------------------
        response = post({"kind": "point", "points": [{"x": 1, "y": 1}]}, frame_index=99)
        check("a frame past the end is 404", response.status_code == 404, response.text)
        response = post(
            {"kind": "point", "points": [{"x": 1, "y": 1}]}, session_id="deadbeefdeadbeef"
        )
        check("an unknown session is 404", response.status_code == 404, response.text)

        # -- the disabled switch ------------------------------------------
        app.dependency_overrides[get_settings] = lambda: Settings(
            temp_dir=TEMP_DIR, enable_sam3=False
        )
        response = client.get("/api/sam3/status")
        body = response.json()
        check(
            "a disabled backend says so in status",
            body["available"] is False and "disabled" in (body["error"] or ""),
            body,
        )
        response = client.post(
            "/api/sam3/segment",
            json={
                "session_id": session.id,
                "frame_index": 0,
                "prompt": {"kind": "point", "points": [{"x": 4, "y": 4}]},
            },
        )
        check("a disabled backend answers 503", response.status_code == 503, response.text)
    finally:
        app.dependency_overrides.clear()
        client.close()
        shutil.rmtree(ROOT, ignore_errors=True)

    print(f"\n{checks} checks passed")


if __name__ == "__main__":
    main()
