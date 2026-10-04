"""HTTP checks for `/api/sam3`, with a stub segmentation service.

The model is replaced by a stub through FastAPI's dependency override, so this
covers what the *route* is responsible for: prompt validation, session lookup,
and the shape of the response. No GPU and no `sam3` install are needed.

The environment is configured by the shared `conftest` before the app is imported
(`get_settings` is cached and `main` resolves settings while being imported), and
`api_client` supplies a client already carrying the `X-Vsr-Client` header every
endpoint under test requires.
"""

import os

import numpy as np
import pytest
from PIL import Image

from src.core.config import Settings, get_settings
from src.core.sessions import create_session, frame_name
from src.domain.prompts import InstanceMask, SegmentResult
from src.domain.rle import decode_rle
from src.inference.registry import get_sam3_service
from src.main import app

HEIGHT, WIDTH = 12, 20

#: Where the stub paints its mask, so the response can be checked against it.
MASK = np.zeros((HEIGHT, WIDTH), dtype=bool)
MASK[3:7, 4:11] = True


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


@pytest.fixture
def stub():
    """The fake service, installed as the app's SAM 3 dependency for one test.

    Installed as a dependency override rather than by monkeypatching, so the route
    exercises its real wiring. Cleared afterwards, so a later test cannot inherit
    the stub and quietly pass against it instead of the code under test.
    """
    service = StubImageService()
    app.dependency_overrides[get_sam3_service] = lambda: service
    yield service
    app.dependency_overrides.clear()


@pytest.fixture
def session(client_id):
    """A three-frame session the app can actually find.

    Created in the *configured* temp dir rather than a per-test one: the route
    opens the session through settings, so a session made anywhere else is a 404.
    """
    session = create_session(get_settings().temp_dir, owner=client_id)
    for index in range(3):
        Image.new("RGB", (WIDTH, HEIGHT), (40, 80 + index * 20, 160)).save(
            os.path.join(session.frames_dir, frame_name(index)), format="JPEG"
        )
    return session


def segment(api_client, session_id, prompt, *, frame_index=0, **options):
    """POST one prompt, with any extra top-level options, and return the response."""
    body = {"session_id": session_id, "frame_index": frame_index, "prompt": prompt}
    body.update(options)
    return api_client.post("/api/sam3/segment", json=body)


def test_status_reports_the_model(api_client, stub):
    response = api_client.get("/api/sam3/status")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == "stub-sam3", body
    assert body["point_prompts"] is True, body


def test_a_point_prompt_round_trips(api_client, stub, session):
    """The click path: a prompt in, the mask and its alternatives out."""
    response = segment(
        api_client,
        session.id,
        {"kind": "point", "points": [{"x": 4, "y": 5}, {"x": 9, "y": 3, "label": 0}]},
        frame_index=1,
    )
    assert response.status_code == 200, response.text
    body = response.json()

    decoded = decode_rle(body["rle"]["size"], body["rle"]["counts"])
    assert np.array_equal(decoded, MASK), decoded.sum()
    assert body["runs"], "runs are sent so the browser can paint without decoding"
    assert body["bbox"] == [4, 3, 7, 4], body["bbox"]
    assert body["area"] == int(MASK.sum()), body["area"]
    assert body["decoder_ms"] == 3.2, body
    assert body["embedding_reused"] is True
    assert "point" in body["prompt"], body["prompt"]
    assert len(body["instances"]) == 2, body["instances"]
    assert body["instance_scores"] == [0.91, 0.42], body["instance_scores"]

    # The route must hand the service what the caller actually sent.
    sent = stub.calls[-1][2]
    assert [point.label for point in sent.points] == [1, 0], sent.points
    assert stub.calls[-1][1] == 1, "the frame index reaches the service"


def test_max_instances_caps_the_alternatives(api_client, stub, session):
    response = segment(
        api_client,
        session.id,
        {"kind": "point", "points": [{"x": 5, "y": 5}]},
        max_instances=1,
    )
    assert response.status_code == 200, response.text
    returned = response.json()["instances"]
    assert returned == [] or len(returned) == 1, returned


def test_a_box_prompt_is_converted(api_client, stub, session):
    response = segment(
        api_client,
        session.id,
        {"kind": "box", "boxes": [{"x0": 2, "y0": 2, "x1": 12, "y1": 9}]},
    )
    assert response.status_code == 200, response.text
    box = stub.calls[-1][2].primary_box
    assert (box.x1, box.y1) == (12.0, 9.0), box
    # The model wants normalised cxcywh, which the prompt converts to on demand.
    assert len(box.to_cxcywh_normalized(WIDTH, HEIGHT)) == 4


def test_a_text_prompt_is_trimmed(api_client, stub, session):
    response = segment(
        api_client, session.id, {"kind": "text", "text": "  shark "}, frame_index=2
    )
    assert response.status_code == 200, response.text
    assert stub.calls[-1][2].text == "shark"


def test_a_frame_past_the_end_is_404(api_client, stub, session):
    response = segment(
        api_client,
        session.id,
        {"kind": "point", "points": [{"x": 1, "y": 1}]},
        frame_index=99,
    )
    assert response.status_code == 404, response.text


def test_an_unknown_session_is_404(api_client, stub):
    response = segment(
        api_client,
        "deadbeefdeadbeef",
        {"kind": "point", "points": [{"x": 1, "y": 1}]},
    )
    assert response.status_code == 404, response.text


#: Each of these is a request a reviewer could plausibly produce, and every one of
#: them must be refused as a bad request rather than reaching a model.
BAD_PROMPTS = [
    ("a point prompt with no click", {"kind": "point", "points": []}),
    ("a click outside the frame", {"kind": "point", "points": [{"x": 99, "y": 3}]}),
    ("a negative coordinate", {"kind": "point", "points": [{"x": -2, "y": 3}]}),
    ("a box prompt with no box", {"kind": "box", "boxes": []}),
    (
        "a degenerate box",
        {"kind": "box", "boxes": [{"x0": 2, "y0": 2, "x1": 2.5, "y1": 9}]},
    ),
    ("a box leaving the frame", {"kind": "box", "boxes": [[0, 0, 500, 500]]}),
    ("an empty text prompt", {"kind": "text", "text": "   "}),
    ("an unknown prompt kind", {"kind": "scribble", "text": "x"}),
    (
        "text mixed with clicks",
        {"kind": "text", "points": [{"x": 1, "y": 1}], "text": "shark"},
    ),
    (
        "clicks mixed with text",
        {"kind": "point", "points": [{"x": 1, "y": 1}], "text": "shark"},
    ),
]


@pytest.mark.parametrize(
    "prompt",
    [prompt for _label, prompt in BAD_PROMPTS],
    ids=[label for label, _prompt in BAD_PROMPTS],
)
def test_a_bad_prompt_is_rejected(api_client, stub, session, prompt):
    response = segment(api_client, session.id, prompt)
    assert response.status_code == 422, response.text


def test_a_disabled_backend_answers_503(api_client, session):
    """`SAM3_ENABLED=0` must say so and refuse, not reach for a model."""
    disabled = Settings(temp_dir=get_settings().temp_dir, enable_sam3=False)
    app.dependency_overrides[get_settings] = lambda: disabled
    try:
        body = api_client.get("/api/sam3/status").json()
        assert body["available"] is False, body
        assert "disabled" in (body["error"] or ""), body

        response = segment(
            api_client,
            session.id,
            {"kind": "point", "points": [{"x": 4, "y": 4}]},
        )
        assert response.status_code == 503, response.text
    finally:
        app.dependency_overrides.pop(get_settings, None)
