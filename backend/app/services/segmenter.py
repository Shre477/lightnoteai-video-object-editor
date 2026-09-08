"""Precise object segmentation.

Primary path: Meta's Segment Anything Model (SAM), prompted with the
bounding box from the detector, giving a pixel-accurate mask.

Fallback: OpenCV GrabCut, used automatically if `segment-anything` isn't
installed (e.g. lighter local setups), or reused per-frame during tracking
where re-running full SAM on every frame would be too slow for a CPU demo.
"""
import numpy as np


def segment_object(frame_bgr, bbox):
    from ..config import settings

    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError:
        return grabcut_fallback(frame_bgr, bbox)

    if not hasattr(segment_object, "_predictor"):
        sam = sam_model_registry[settings.SAM_MODEL_TYPE](checkpoint=settings.SAM_CHECKPOINT)
        segment_object._predictor = SamPredictor(sam)

    predictor = segment_object._predictor
    predictor.set_image(frame_bgr[:, :, ::-1])
    x, y, w, h = bbox
    box = np.array([x, y, x + w, y + h])
    masks, scores, _ = predictor.predict(box=box, multimask_output=True)
    return masks[int(np.argmax(scores))].astype(np.uint8)


def grabcut_fallback(frame_bgr, bbox):
    import cv2

    x, y, w, h = bbox
    h_frame, w_frame = frame_bgr.shape[:2]
    # Clamp box inside the frame - trackers can drift slightly out of bounds.
    x = max(0, min(x, w_frame - 1))
    y = max(0, min(y, h_frame - 1))
    w = max(1, min(w, w_frame - x))
    h = max(1, min(h, h_frame - y))

    mask = np.zeros(frame_bgr.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(frame_bgr, mask, (x, y, w, h), bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
        return np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
    except Exception:
        # GrabCut can fail on degenerate boxes; fall back to a solid box mask.
        m = np.zeros(frame_bgr.shape[:2], np.uint8)
        m[y:y + h, x:x + w] = 1
        return m
