const API_BASE = "http://localhost:8000/api";

const form = document.getElementById("job-form");
const startBtn = document.getElementById("start-btn");
const statusSection = document.getElementById("status-section");
const progressFill = document.getElementById("progress-fill");
const statusText = document.getElementById("status-text");
const parsedWrap = document.getElementById("parsed-instruction-wrap");
const parsedInstructionEl = document.getElementById("parsed-instruction");
const resultSection = document.getElementById("result-section");
const resultVideo = document.getElementById("result-video");
const downloadLink = document.getElementById("download-link");
const errorText = document.getElementById("error-text");

let pollTimer = null;

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorText.hidden = true;
  resultSection.hidden = true;
  parsedWrap.hidden = true;

  const video = document.getElementById("video").files[0];
  const videoUrl = document.getElementById("video-url").value.trim();
  const reference = document.getElementById("reference").files[0];
  const prompt = document.getElementById("prompt").value.trim();

  if (!video && !videoUrl) {
    showError("Please upload a video file or paste a video URL.");
    return;
  }

  const fd = new FormData();
  fd.append("prompt", prompt);
  if (video) fd.append("video", video);
  if (videoUrl) fd.append("video_url", videoUrl);
  if (reference) fd.append("reference_image", reference);

  startBtn.disabled = true;
  startBtn.textContent = "Uploading...";

  try {
    const res = await fetch(`${API_BASE}/jobs`, { method: "POST", body: fd });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Failed to start job");
    }
    const { job_id } = await res.json();
    statusSection.hidden = false;
    pollStatus(job_id);
  } catch (err) {
    showError(err.message);
    resetButton();
  }
});

function pollStatus(jobId) {
  clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    try {
      const res = await fetch(`${API_BASE}/jobs/${jobId}`);
      const job = await res.json();

      progressFill.style.width = `${job.progress}%`;
      statusText.textContent = `${job.status} - ${job.message || ""}`;

      if (job.parsed_instruction) {
        parsedWrap.hidden = false;
        parsedInstructionEl.textContent = JSON.stringify(job.parsed_instruction, null, 2);
      }

      if (job.status === "completed") {
        clearInterval(pollTimer);
        const url = `${API_BASE}/jobs/${jobId}/result`;
        resultVideo.src = url;
        downloadLink.href = url;
        resultSection.hidden = false;
        resetButton();
      }

      if (job.status === "failed") {
        clearInterval(pollTimer);
        showError(job.error || "Processing failed");
        resetButton();
      }
    } catch (err) {
      clearInterval(pollTimer);
      showError("Lost connection to the backend.");
      resetButton();
    }
  }, 1500);
}

function resetButton() {
  startBtn.disabled = false;
  startBtn.textContent = "Start Processing";
}

function showError(msg) {
  errorText.hidden = false;
  errorText.textContent = msg;
}
