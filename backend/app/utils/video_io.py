"""Frame extraction / reassembly. Uses OpenCV for frame I/O and ffmpeg (via
subprocess) only for muxing the original audio track back onto the edited
video, since OpenCV's VideoWriter doesn't handle audio."""
import os
import subprocess

import cv2


def extract_frames(video_path, frames_dir=None):
    if frames_dir:
        os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def _has_audio_stream(video_path: str) -> bool:
    try:
        result = subprocess.run(
            ["ffprobe", "-i", video_path, "-show_streams", "-select_streams", "a", "-loglevel", "error"],
            capture_output=True, text=True, timeout=15,
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def frames_to_video(frames, fps, out_path, source_video_for_audio=None):
    if not frames:
        raise ValueError("No frames to write - the pipeline produced an empty video.")

    h, w = frames[0].shape[:2]
    tmp_path = out_path + ".noaudio.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()

    if source_video_for_audio and _has_audio_stream(source_video_for_audio):
        cmd = [
            "ffmpeg", "-y", "-i", tmp_path, "-i", source_video_for_audio,
            "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0?",
            "-shortest", out_path,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=120)
            os.remove(tmp_path)
            return out_path
        except Exception:
            pass  # fall back to the video-only file below

    os.replace(tmp_path, out_path)
    return out_path
