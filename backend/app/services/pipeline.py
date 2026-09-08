"""Orchestrates the full edit: parse instruction -> detect -> segment ->
track -> remove -> replace -> render. Runs in a background thread per job
and reports progress into the job store so the frontend can poll it.

Two execution modes (see config.DEMO_MODE):
  - DEMO_MODE=true:  a fast, dependency-free stub for detection/segmentation
    (fixed center box). Everything else (instruction parsing, inpainting,
    compositing, rendering) is the real logic. This guarantees the app is
    demoable end-to-end even with no GPU / no model downloads available.
  - DEMO_MODE=false: the real AI path - OWL-ViT for open-vocabulary
    detection, SAM for segmentation, OpenCV CSRT for tracking.
"""
import os

import cv2
import numpy as np

from ..config import settings
from ..job_store import job_store
from ..models.job import JobStatus
from ..utils.video_io import extract_frames, frames_to_video
from .compositor import composite_replacement, generate_replacement_image
from .inpainter import inpaint_frame
from .instruction_parser import parse_instruction


def _demo_detect(frame, target_text):
    h, w = frame.shape[:2]
    bw, bh = w // 3, h // 3
    return (w // 2 - bw // 2, h // 2 - bh // 2, bw, bh)


def _box_mask(frame_shape, bbox):
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    x, y, w, h = bbox
    mask[y:y + h, x:x + w] = 1
    return mask


def run_pipeline(job_id: str) -> None:
    job = job_store.get(job_id)
    if job is None:
        return

    try:
        # 1. Understand the instruction
        job_store.update(job_id, status=JobStatus.PARSING_INSTRUCTION, progress=5,
                          message="Understanding the instruction...")
        intent = parse_instruction(job.prompt)
        job_store.update(job_id, parsed_instruction=intent)

        # 2. Read the video
        frames, fps = extract_frames(job.video_path, os.path.join(settings.FRAMES_DIR, job_id))
        if not frames:
            raise RuntimeError("Could not read any frames from the uploaded video.")

        # 3. Locate the target object (first frame)
        job_store.update(job_id, status=JobStatus.DETECTING_OBJECT, progress=15,
                          message=f"Locating '{intent.get('target')}' in the video...")
        if settings.DEMO_MODE:
            bbox = _demo_detect(frames[0], intent.get("target", ""))
        else:
            from .detector import detect_object
            bbox = detect_object(frames[0], intent.get("target", ""))

        # 4. Segment it precisely + set up tracking for later frames
        job_store.update(job_id, status=JobStatus.SEGMENTING, progress=30,
                          message="Segmenting the object...")
        tracker = None
        if settings.DEMO_MODE:
            mask0 = _box_mask(frames[0].shape, bbox)
        else:
            from .segmenter import segment_object
            from .tracker import BoxTracker
            mask0 = segment_object(frames[0], bbox)
            tracker = BoxTracker(frames[0], bbox)

        # 5. Prepare the replacement image, if this is a replace operation
        replacement_img = None
        if intent.get("operation") == "replace_object":
            if job.reference_image_path and os.path.exists(job.reference_image_path):
                replacement_img = cv2.imread(job.reference_image_path)
            if replacement_img is None:
                replacement_img = generate_replacement_image(
                    intent.get("replacement") or intent.get("target") or ""
                )

        # 6. Edit every frame: remove the original object, then (if
        #    applicable) composite the replacement, tracking the object's
        #    position as it moves through the video.
        job_store.update(job_id, status=JobStatus.EDITING_FRAMES, progress=45,
                          message="Editing frames...")
        edited_frames = []
        current_mask = mask0
        current_bbox = bbox
        total = len(frames)

        for i, frame in enumerate(frames):
            if i > 0:
                if settings.DEMO_MODE:
                    current_mask = _box_mask(frame.shape, current_bbox)
                else:
                    tracked_bbox = tracker.update(frame)
                    if tracked_bbox is not None:
                        current_bbox = tracked_bbox
                    from .segmenter import grabcut_fallback
                    current_mask = grabcut_fallback(frame, current_bbox)

            out_frame = inpaint_frame(frame, current_mask)
            if intent.get("operation") == "replace_object" and replacement_img is not None:
                out_frame = composite_replacement(out_frame, current_mask, replacement_img)
            edited_frames.append(out_frame)

            job_store.update(job_id, progress=45 + int(40 * (i + 1) / total))

        # 7. Render the final video (re-attaching original audio if present)
        job_store.update(job_id, status=JobStatus.RENDERING, progress=90,
                          message="Rendering final video...")
        os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(settings.OUTPUT_DIR, f"{job_id}.mp4")
        frames_to_video(edited_frames, fps, out_path, source_video_for_audio=job.video_path)

        job_store.update(job_id, status=JobStatus.COMPLETED, progress=100,
                          message="Done.", output_path=out_path)

    except Exception as exc:  # noqa: BLE001 - we want to surface *any* failure to the user
        job_store.update(job_id, status=JobStatus.FAILED, message=str(exc), error=str(exc))
