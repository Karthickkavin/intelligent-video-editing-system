"""
API routes for the Autonomous Smart Video Editor Flask backend.

Endpoints:
  POST /api/upload          – upload a video file
  POST /api/process/<job_id> – start processing
  GET  /api/status/<job_id>  – poll job status
  GET  /api/download/<job_id> – download result
  DELETE /api/job/<job_id>   – clean up job files
"""

import os
import threading
import uuid
from typing import Dict, Any

from flask import Blueprint, current_app, jsonify, request, send_file

from core.video_processor import VideoProcessor
from core.decision_engine import DecisionEngine
from core.renderer import Renderer
from utils.validators import allowed_video_file, validate_processing_params
from utils.helpers import generate_unique_filename, safe_delete
from utils.logger import get_logger

logger = get_logger(__name__)

api_bp = Blueprint("api", __name__, url_prefix="/api")

# In-memory job store  { job_id: { status, progress, message, output_path, ... } }
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _update_job(job_id: str, **kwargs) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@api_bp.route("/upload", methods=["POST"])
def upload():
    """Accept a video file and store it in the upload directory."""
    if "file" not in request.files:
        return jsonify({"error": "No file field in request"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No file selected"}), 400

    if not allowed_video_file(f.filename):
        return jsonify({"error": "Unsupported file format"}), 400

    ext = f.filename.rsplit(".", 1)[-1].lower()
    job_id = uuid.uuid4().hex
    filename = f"{job_id}.{ext}"
    upload_path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)

    os.makedirs(current_app.config["UPLOAD_FOLDER"], exist_ok=True)
    f.save(upload_path)
    logger.info("Uploaded file saved: %s (job %s)", upload_path, job_id)

    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "uploaded",
            "progress": 0,
            "message": "File uploaded successfully",
            "input_path": upload_path,
            "output_path": None,
        }

    return jsonify({"job_id": job_id, "message": "File uploaded successfully"}), 201


@api_bp.route("/process/<job_id>", methods=["POST"])
def process(job_id: str):
    """Start background processing for an uploaded video."""
    with _jobs_lock:
        job = _jobs.get(job_id)

    if not job:
        return jsonify({"error": "Job not found"}), 404

    if job["status"] not in ("uploaded", "failed"):
        return jsonify({"error": f"Job is already {job['status']}"}), 409

    # Optional processing params from JSON body
    params = request.get_json(silent=True) or {}
    error_msg = validate_processing_params(params)
    if error_msg:
        return jsonify({"error": error_msg}), 400

    _update_job(job_id, status="queued", progress=0, message="Job queued")

    thread = threading.Thread(
        target=_run_processing,
        args=(job_id, job["input_path"], params),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id, "message": "Processing started"}), 202


@api_bp.route("/status/<job_id>", methods=["GET"])
def status(job_id: str):
    """Return the current status of a job."""
    with _jobs_lock:
        job = _jobs.get(job_id)

    if not job:
        return jsonify({"error": "Job not found"}), 404

    return jsonify(
        {
            "job_id": job_id,
            "status": job["status"],
            "progress": job["progress"],
            "message": job["message"],
        }
    )


@api_bp.route("/download/<job_id>", methods=["GET"])
def download(job_id: str):
    """Download the processed output video."""
    with _jobs_lock:
        job = _jobs.get(job_id)

    if not job:
        return jsonify({"error": "Job not found"}), 404

    if job["status"] != "completed":
        return jsonify({"error": "Job is not yet completed"}), 409

    output_path = job.get("output_path")
    if not output_path or not os.path.exists(output_path):
        return jsonify({"error": "Output file not found"}), 404

    return send_file(
        output_path,
        as_attachment=True,
        download_name=f"edited_{job_id}.mp4",
        mimetype="video/mp4",
    )


@api_bp.route("/job/<job_id>", methods=["DELETE"])
def delete_job(job_id: str):
    """Clean up job files and remove job record."""
    with _jobs_lock:
        job = _jobs.pop(job_id, None)

    if not job:
        return jsonify({"error": "Job not found"}), 404

    safe_delete(job.get("input_path", ""))
    safe_delete(job.get("output_path", ""))

    return jsonify({"message": "Job deleted"})


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def _run_processing(job_id: str, input_path: str, params: dict) -> None:
    """Run the full processing pipeline in a background thread."""
    try:
        base_cfg = current_app.config["PROCESSOR_CONFIG"]

        # Create a per-job copy of config to avoid race conditions in multi-threaded use
        import copy
        cfg = copy.copy(base_cfg)

        # Apply per-request parameter overrides on the copy
        if "silence_threshold" in params:
            cfg.SILENCE_THRESHOLD = float(params["silence_threshold"])
        if "min_segment_score" in params:
            cfg.MIN_SEGMENT_SCORE = float(params["min_segment_score"])
        if "min_segment_duration" in params:
            cfg.MIN_SEGMENT_DURATION = float(params["min_segment_duration"])

        _update_job(job_id, status="processing", progress=10, message="Analysing video")

        processor = VideoProcessor(cfg)
        segments = processor.analyze(input_path)

        _update_job(job_id, progress=40, message="Scoring segments")

        engine = DecisionEngine(cfg)
        final_segments = engine.process(segments)

        if not final_segments:
            _update_job(
                job_id,
                status="failed",
                progress=0,
                message="No suitable segments found in video",
            )
            return

        _update_job(job_id, progress=60, message=f"Rendering {len(final_segments)} segments")

        output_filename = generate_unique_filename(prefix="output", extension="mp4")
        output_path = os.path.join(current_app.config["OUTPUT_FOLDER"], output_filename)
        os.makedirs(current_app.config["OUTPUT_FOLDER"], exist_ok=True)

        renderer = Renderer(cfg)
        renderer.render(input_path, final_segments, output_path)

        _update_job(
            job_id,
            status="completed",
            progress=100,
            message="Processing complete",
            output_path=output_path,
        )
        logger.info("Job %s completed → %s", job_id, output_path)

    except Exception as exc:  # noqa: BLE001
        logger.error("Job %s failed: %s", job_id, exc, exc_info=True)
        _update_job(
            job_id,
            status="failed",
            progress=0,
            message=str(exc),
        )
