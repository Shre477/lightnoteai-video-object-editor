"""Places a replacement object into the region the original occupied.

Two sources for the replacement image:
  - A user-supplied reference image (preferred - most faithful to the ask).
  - An AI-generated image, when no reference image was given, produced from
    the "replacement" text via an image-generation API.
"""
import cv2
import numpy as np


def composite_replacement(frame_bgr, mask, replacement_img_bgr):
    mask_bin = mask.astype(np.uint8)
    if mask_bin.max() > 1:
        mask_bin = (mask_bin > 127).astype(np.uint8)

    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0:
        return frame_bgr

    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return frame_bgr

    resized = cv2.resize(replacement_img_bgr, (w, h))
    roi_mask = mask_bin[y1:y2, x1:x2].astype(np.float32)
    # Feather the mask edges so the paste blends rather than looking pasted.
    roi_mask = cv2.GaussianBlur(roi_mask, (9, 9), 0)
    roi_mask_3c = np.repeat(roi_mask[:, :, None], 3, axis=2)

    out = frame_bgr.copy()
    region = out[y1:y2, x1:x2].astype(np.float32)
    blended = (resized.astype(np.float32) * roi_mask_3c + region * (1 - roi_mask_3c))
    out[y1:y2, x1:x2] = blended.astype(np.uint8)
    return out


def generate_replacement_image(description: str, size=(512, 512)):
    """Used only when the user did not upload a reference image. Requires
    OPENAI_API_KEY; otherwise returns a clearly-labelled placeholder so the
    pipeline still completes end-to-end instead of failing the whole job."""
    import cv2
    import numpy as np
    import requests

    from ..config import settings

    if settings.OPENAI_API_KEY:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            resp = client.images.generate(
                model="dall-e-3", prompt=description, size="1024x1024", n=1
            )
            url = resp.data[0].url
            img_bytes = requests.get(url, timeout=30).content
            arr = np.frombuffer(img_bytes, np.uint8)
            decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if decoded is not None:
                return decoded
        except Exception:
            pass  # fall through to placeholder

    placeholder = np.full((size[1], size[0], 3), (60, 60, 200), dtype=np.uint8)
    cv2.putText(
        placeholder, description[:22], (10, size[1] // 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )
    return placeholder
