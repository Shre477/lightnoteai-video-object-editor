import os
import shutil
import threading
import uuid
from typing import Optional

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .config import settings
from .job_store import job_store
from .models.job import Job, JobStatus
from .services.pipeline import run_pipeline

app = FastAPI(title="LightNoteAI Video Editor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.ALLOWED_ORIGINS] if settings.ALLOWED_ORIGINS != "*" else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)


@app.get("/api/health")
def health():
    return {"status": "ok", "demo_mode": settings.DEMO_MODE}


@app.post("/api/jobs")
async def create_job(
    prompt: str = Form(...),
    video: Optional[UploadFile] = File(None),
    video_url: Optional[str] = Form(None),
    reference_image: Optional[UploadFile] = File(None),
):
    if not video and not video_url:
        raise HTTPException(400, "Provide either a video file or a video_url.")

    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(settings.UPLOAD_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    # Video source: uploaded file takes priority; otherwise download the URL
    # (bonus feature: "Video URL Input").
    if video:
        video_path = os.path.join(job_dir, video.filename or "input.mp4")
        with open(video_path, "wb") as f:
            shutil.copyfileobj(video.file, f)
    else:
        video_path = os.path.join(job_dir, "input.mp4")
        try:
            resp = requests.get(video_url, stream=True, timeout=30)
            resp.raise_for_status()
            with open(video_path, "wb") as f:
                shutil.copyfileobj(resp.raw, f)
        except Exception as exc:
            raise HTTPException(400, f"Could not download video_url: {exc}")

    ref_path = None
    if reference_image is not None:
        ref_path = os.path.join(job_dir, reference_image.filename or "reference.png")
        with open(ref_path, "wb") as f:
            shutil.copyfileobj(reference_image.file, f)

    job = Job(id=job_id, video_path=video_path, reference_image_path=ref_path, prompt=prompt)
    job_store.add(job)

    thread = threading.Thread(target=run_pipeline, args=(job_id,), daemon=True)
    thread.start()

    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return {
        "id": job.id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "parsed_instruction": job.parsed_instruction,
        "error": job.error,
    }


@app.get("/api/jobs/{job_id}/result")
def get_result(job_id: str):
    job = job_store.get(job_id)
    if job is None or job.status != JobStatus.COMPLETED or not job.output_path:
        raise HTTPException(404, "Result not ready")
    return FileResponse(job.output_path, media_type="video/mp4", filename=f"{job_id}.mp4")
