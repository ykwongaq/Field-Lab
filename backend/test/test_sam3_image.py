"""Checks for the image service's per-reviewer embedding cache.

No model and no GPU are involved: a stub manager and processor stand in for SAM 3
and count how often the vision backbone runs. That count is the whole point of the
cache — a hit skips the encode — so it is worth pinning down before there is a
GPU to try it on. (It is also the first thing in this suite to execute
`Sam3ImageService.segment`'s structure: the cache, the state copy and the
instance normalisation, all without weights.)

The property under test is *fairness*, not just hit rate. A single flat LRU is
less code and is wrong for a shared server: one reviewer's clicks evict another's
frames, and every eviction costs a full backbone pass, so two people working at
once would encode roughly twice as often as the same two working alone.
"""

import contextlib
import os
from types import SimpleNamespace

import numpy as np
from PIL import Image
from src.core.sessions import create_session, frame_name
from src.domain.prompts import KIND_TEXT, SegmentPrompt
from src.inference.gate import ModelGate
from src.inference.sam3_image import Sam3ImageService

HEIGHT, WIDTH = 12, 20
ALICE = "a" * 32
BOB = "b" * 32


class StubProcessor:
    """Stands in for `Sam3Processor`, counting vision-backbone passes."""

    def __init__(self):
        self.encodes = 0

    def set_image(self, image):
        self.encodes += 1
        return {
            "original_height": image.height,
            "original_width": image.width,
            "backbone_out": {"features": "encoded"},
        }

    def set_text_prompt(self, _text, _state):
        masks = np.zeros((1, HEIGHT, WIDTH), dtype=bool)
        masks[0, 3:7, 4:11] = True
        return {"masks": masks, "scores": np.array([0.9])}


class StubManager:
    """The slice of `ModelManager` that `segment()` touches."""

    def __init__(self, processor):
        self.gate = ModelGate()
        self.config = SimpleNamespace(enable_inst_interactivity=True)
        self._processor = processor

    def status(self):
        return {
            "available": True,
            "loaded": True,
            "loaded_models": ["image"],
            "model": "stub",
            "device": "cpu",
            "error": None,
        }

    @contextlib.contextmanager
    def inference_context(self):
        yield

    def image(self):
        return (SimpleNamespace(), self._processor)


class FakeTensor:
    """A stand-in for a CUDA torch tensor: numpy refuses it until it is moved."""

    def __init__(self, array):
        self._array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class TensorProcessor(StubProcessor):
    """Like `StubProcessor`, but hands back CUDA-like tensors as the models do."""

    def set_text_prompt(self, text, state):
        payload = super().set_text_prompt(text, state)
        return {
            "masks": FakeTensor(payload["masks"]),
            "scores": FakeTensor(payload["scores"]),
        }


class Bfloat16Tensor:
    """A stand-in for an autocast bf16 tensor: numpy refuses it until widened."""

    def __init__(self, array):
        self._array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        raise TypeError("Got unsupported ScalarType BFloat16")

    def float(self):
        return FakeTensor(self._array.astype(np.float32))


class Bfloat16Processor(StubProcessor):
    """Like `StubProcessor`, but hands back bf16 tensors, as autocast makes SAM 3."""

    def set_text_prompt(self, text, state):
        payload = super().set_text_prompt(text, state)
        return {
            "masks": Bfloat16Tensor(payload["masks"].astype(np.float32)),
            "scores": Bfloat16Tensor(payload["scores"].astype(np.float32)),
        }


class Rig:
    """A service over its stub processor, plus a way to make sessions."""

    def __init__(self, root, *, cache_size=2, cache_clients=2, processor=None):
        self.root = root
        self.processor = processor or StubProcessor()
        self.service = Sam3ImageService(
            manager=StubManager(self.processor),
            cache_size=cache_size,
            cache_clients=cache_clients,
        )

    def session(self, owner, tag):
        """A session owned by `owner`, holding three tiny real frames."""
        session = create_session(self.root, owner=owner)
        for index in range(3):
            Image.new("RGB", (WIDTH, HEIGHT), (index * 40, 60, 120)).save(
                os.path.join(session.frames_dir, frame_name(index)), format="JPEG"
            )
        session.update_meta(tag=tag)
        return session

    def click(self, session, frame_index):
        """One text prompt on one frame; `True` when the encode was skipped."""
        result = self.service.segment(
            session, frame_index, SegmentPrompt.from_wire(KIND_TEXT, text="coral")
        )
        return result.embedding_reused


def test_two_reviewers_do_not_evict_each_other(scratch):
    """Alice's second visit to her own frame is still a hit while Bob works."""
    rig = Rig(scratch, cache_size=2, cache_clients=2)
    alice = rig.session(ALICE, "alice")
    bob = rig.session(BOB, "bob")

    assert rig.click(alice, 0) is False
    assert rig.click(alice, 1) is False
    assert rig.click(bob, 0) is False
    assert rig.click(bob, 1) is False
    assert rig.processor.encodes == 4, "each reviewer encodes their frames once"

    # Frame 0 is still Alice's even though Bob has been clicking since.
    assert (
        rig.click(alice, 0) is True
    ), "alice's cached frame was evicted by bob's clicks"
    assert rig.processor.encodes == 4, rig.processor.encodes


def test_one_reviewer_is_still_capped(scratch):
    """The per-reviewer budget is a real cap, not an unbounded cache."""
    rig = Rig(scratch, cache_size=2, cache_clients=2)
    session = rig.session(ALICE, "alice")

    for index in range(3):
        assert rig.click(session, index) is False
    assert rig.processor.encodes == 3

    # Frame 0 is the oldest of three in a two-frame budget, so it has gone.
    assert rig.click(session, 0) is False, "frame 0 should have been evicted"
    assert rig.processor.encodes == 4

    # ...and frame 2, the most recent, is still cached.
    assert rig.click(session, 2) is True
    assert rig.processor.encodes == 4


def test_the_reviewer_count_is_bounded(scratch):
    """With room for one reviewer, a second one takes the cache away."""
    rig = Rig(scratch, cache_size=2, cache_clients=1)
    alice = rig.session(ALICE, "alice")
    bob = rig.session(BOB, "bob")

    assert rig.click(alice, 0) is False
    assert rig.click(bob, 0) is False
    assert rig.click(alice, 0) is False, "alice's cache should have been dropped"
    assert rig.processor.encodes == 3, rig.processor.encodes


def test_status_counts_frames_not_reviewers(scratch):
    """`cache_entries` means frames, which is what the status chip reports."""
    rig = Rig(scratch, cache_size=2, cache_clients=2)
    alice = rig.session(ALICE, "alice")
    bob = rig.session(BOB, "bob")

    rig.click(alice, 0)
    rig.click(bob, 0)
    reported = rig.service.status()
    assert reported["cache_entries"] == 2, reported


def test_a_gpu_tensor_result_is_moved_to_the_host(scratch):
    """The real models return CUDA tensors, and numpy refuses those outright.

    A regression check for a failure only a live model could reveal: `np.asarray`
    on a CUDA tensor raises ("Use Tensor.cpu() to copy the tensor to host memory
    first") rather than copying, so every prompt answered 500 until the result was
    moved to the host before numpy saw it.
    """
    rig = Rig(scratch, cache_size=2, cache_clients=1, processor=TensorProcessor())
    session = rig.session(ALICE, "tensor")
    result = rig.service.segment(
        session, 0, SegmentPrompt.from_wire(KIND_TEXT, text="coral")
    )
    # The stub paints rows 3:7 of columns 4:11, so 4 x 7 pixels are set.
    assert len(result.instances) == 1, result.instances
    assert result.area == 28, result.area


def test_a_bfloat16_result_is_widened_for_numpy(scratch):
    """Autocast bf16 outputs are unreadable by numpy until widened to fp32.

    A regression check for "TypeError: Got unsupported ScalarType BFloat16": SAM 3
    runs under `torch.autocast(dtype=bfloat16)`, so a text prompt's scores come
    back bf16 and `.numpy()` raises instead of copying. `_to_numpy` has to widen
    them to float32, rather than letting every text prompt answer 500.
    """
    rig = Rig(scratch, cache_size=2, cache_clients=1, processor=Bfloat16Processor())
    session = rig.session(ALICE, "bfloat16")
    result = rig.service.segment(
        session, 0, SegmentPrompt.from_wire(KIND_TEXT, text="coral")
    )
    assert len(result.instances) == 1, result.instances
    assert result.area == 28, result.area
