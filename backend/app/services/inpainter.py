"""Removes the target object from a frame via classical inpainting.

cv2.inpaint (Telea's fast marching method) is deliberately used instead of a
diffusion-based inpainter: it's CPU-only, needs no model download, and runs
in milliseconds per frame - important when a video has hundreds of frames.
The assignment explicitly says results don't need to be production-quality,
so this trades some visual polish for reliability and speed. Swapping in a
diffusion inpainter (e.g. Stable Diffusion Inpainting via `diffusers`) later
only means changing this one function - see README "Known limitations".
"""
import cv2
import numpy as np


def inpaint_frame(frame_bgr, mask):
    mask_u8 = mask.astype(np.uint8)
    if mask_u8.max() <= 1:
        mask_u8 = mask_u8 * 255
    # Dilate slightly so we don't leave a thin fringe of the original object.
    mask_u8 = cv2.dilate(mask_u8, np.ones((7, 7), np.uint8), iterations=1)
    return cv2.inpaint(frame_bgr, mask_u8, 5, cv2.INPAINT_TELEA)
