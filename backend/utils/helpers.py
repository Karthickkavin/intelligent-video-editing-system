"""
General-purpose helper utilities for the Autonomous Smart Video Editor.
"""

import os
import uuid
import subprocess
from typing import Optional


def generate_unique_filename(prefix: str = "video", extension: str = "mp4") -> str:
    """Generate a unique filename using a UUID."""
    return f"{prefix}_{uuid.uuid4().hex}.{extension}"


def ensure_directory(path: str) -> None:
    """Create the directory (and any missing parents) if it does not exist."""
    os.makedirs(path, exist_ok=True)


def format_duration(seconds: float) -> str:
    """Convert a duration in seconds to HH:MM:SS format."""
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def ffmpeg_available() -> bool:
    """Return True if FFmpeg is available on PATH."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def ffprobe_available() -> bool:
    """Return True if FFprobe is available on PATH."""
    try:
        subprocess.run(
            ["ffprobe", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def safe_delete(filepath: str) -> None:
    """Delete a file without raising an exception if it does not exist."""
    try:
        os.remove(filepath)
    except FileNotFoundError:
        pass


def get_file_size_mb(filepath: str) -> Optional[float]:
    """Return file size in megabytes, or None if the file does not exist."""
    try:
        return os.path.getsize(filepath) / (1024 * 1024)
    except OSError:
        return None
