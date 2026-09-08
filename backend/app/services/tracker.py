"""Propagates a bounding box across frames without re-running the (expensive)
detector on every single frame - a pragmatic "suitable approach" per the
assignment brief for tracking the object through the video.
"""
import cv2


def _create_tracker():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    # Some OpenCV builds ship legacy trackers under cv2.legacy
    return cv2.legacy.TrackerCSRT_create()


class BoxTracker:
    def __init__(self, first_frame, bbox):
        self.tracker = _create_tracker()
        self.tracker.init(first_frame, tuple(int(v) for v in bbox))

    def update(self, frame):
        ok, bbox = self.tracker.update(frame)
        if not ok:
            return None
        return tuple(int(v) for v in bbox)
