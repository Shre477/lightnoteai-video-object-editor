import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class JobStatus(str, Enum):
    QUEUED = "queued"
    PARSING_INSTRUCTION = "parsing_instruction"
    DETECTING_OBJECT = "detecting_object"
    SEGMENTING = "segmenting"
    EDITING_FRAMES = "editing_frames"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    video_path: str = ""
    reference_image_path: Optional[str] = None
    prompt: str = ""
    status: JobStatus = JobStatus.QUEUED
    progress: int = 0
    message: str = ""
    parsed_instruction: Optional[dict] = None
    output_path: Optional[str] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
