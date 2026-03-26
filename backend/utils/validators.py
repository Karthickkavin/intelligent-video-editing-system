"""
Input validation utilities for the Autonomous Smart Video Editor.
"""

import os
from typing import Optional


ALLOWED_EXTENSIONS = {"mp4", "mov", "mkv", "avi", "webm"}
MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB


def allowed_video_file(filename: str) -> bool:
    """Return True if the filename has an allowed video extension."""
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def validate_file_size(filepath: str, max_bytes: int = MAX_FILE_SIZE_BYTES) -> bool:
    """Return True if the file size is within the allowed limit."""
    return os.path.getsize(filepath) <= max_bytes


def validate_processing_params(params: dict) -> Optional[str]:
    """
    Validate processing parameters supplied by the caller.

    Returns an error message string if invalid, or None if all params are valid.
    """
    silence_threshold = params.get("silence_threshold")
    if silence_threshold is not None:
        try:
            val = float(silence_threshold)
            if not (-80 <= val <= 0):
                return "silence_threshold must be between -80 and 0 dB"
        except (TypeError, ValueError):
            return "silence_threshold must be a number"

    min_score = params.get("min_segment_score")
    if min_score is not None:
        try:
            val = float(min_score)
            if not (0.0 <= val <= 1.0):
                return "min_segment_score must be between 0.0 and 1.0"
        except (TypeError, ValueError):
            return "min_segment_score must be a number"

    min_duration = params.get("min_segment_duration")
    if min_duration is not None:
        try:
            val = float(min_duration)
            if val <= 0:
                return "min_segment_duration must be positive"
        except (TypeError, ValueError):
            return "min_segment_duration must be a number"

    return None
