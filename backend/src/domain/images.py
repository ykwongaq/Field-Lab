"""Image decoding helpers shared by the inference services.

`cv2` and `PIL` are imported inside the functions so that importing the package
(and therefore the settings, schemas and pure RLE logic) never requires the
image stack.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.core.errors import InvalidRequest


def decode_image_rgb(data: bytes) -> np.ndarray:
    """JPEG/PNG bytes -> RGB uint8 array (H, W, 3)."""
    import cv2

    buffer = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if bgr is None:
        raise InvalidRequest("The frame could not be decoded as an image.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def decode_image_pil(data: bytes) -> Any:
    """JPEG/PNG bytes -> PIL image in RGB (SAM 3 expects PIL input)."""
    from PIL import Image

    return Image.fromarray(decode_image_rgb(data))
