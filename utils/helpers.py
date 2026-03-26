"""
Common utility helpers shared across all modules.
"""

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("video_editor.helpers")


# ---------------------------------------------------------------------------
# File / path utilities
# ---------------------------------------------------------------------------


def ensure_dir(path: str) -> str:
    """Create *path* (and parents) if it does not already exist.

    Returns the (possibly newly created) path string.
    """
    os.makedirs(path, exist_ok=True)
    return path


def cleanup_temp_files(temp_dir: str) -> None:
    """Remove all files inside *temp_dir* (keeps the directory itself)."""
    if not os.path.isdir(temp_dir):
        return
    for entry in os.scandir(temp_dir):
        try:
            if entry.is_file() or entry.is_symlink():
                os.remove(entry.path)
            elif entry.is_dir():
                shutil.rmtree(entry.path)
        except OSError as exc:
            logger.warning("Could not remove %s: %s", entry.path, exc)


# ---------------------------------------------------------------------------
# Video file validation
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"}


def validate_video_file(file_path: str, max_size_mb: float = 2048.0) -> Tuple[bool, str]:
    """Validate that *file_path* is a readable, supported video file.

    Parameters
    ----------
    file_path:
        Absolute or relative path to check.
    max_size_mb:
        Maximum allowed file size in megabytes.

    Returns
    -------
    (ok, message)
        ``ok`` is ``True`` when the file passes all checks.
        ``message`` explains what went wrong when ``ok`` is ``False``.
    """
    path = Path(file_path)

    if not path.exists():
        return False, f"File not found: {file_path}"

    if not path.is_file():
        return False, f"Path is not a file: {file_path}"

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return False, (
            f"Unsupported format '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_size_mb:
        return False, (
            f"File size {size_mb:.1f} MB exceeds the limit of {max_size_mb:.0f} MB."
        )

    return True, "OK"


# ---------------------------------------------------------------------------
# Video metadata
# ---------------------------------------------------------------------------


def get_video_info(file_path: str) -> Dict:
    """Return a dictionary of basic video metadata.

    Uses OpenCV when available; falls back to a minimal dict otherwise.
    """
    info: Dict = {
        "path": file_path,
        "filename": os.path.basename(file_path),
        "size_mb": 0.0,
        "duration": 0.0,
        "fps": 0.0,
        "frame_count": 0,
        "width": 0,
        "height": 0,
        "codec": "unknown",
    }

    try:
        size_bytes = os.path.getsize(file_path)
        info["size_mb"] = size_bytes / (1024 * 1024)
    except OSError:
        pass

    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(file_path)
        if cap.isOpened():
            info["fps"] = cap.get(cv2.CAP_PROP_FPS) or 0.0
            info["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if info["fps"] > 0:
                info["duration"] = info["frame_count"] / info["fps"]
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            info["codec"] = "".join(
                chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)
            ).strip()
        cap.release()
    except Exception as exc:  # noqa: BLE001
        logger.warning("OpenCV video info failed for %s: %s", file_path, exc)

    return info


# ---------------------------------------------------------------------------
# Time formatting
# ---------------------------------------------------------------------------


def format_time(seconds: float) -> str:
    """Convert *seconds* to a human-readable ``HH:MM:SS`` string."""
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------


class ProgressTracker:
    """Simple progress tracker that prints percentage updates to stdout."""

    def __init__(self, total: int, description: str = "Processing") -> None:
        self.total = max(1, total)
        self.description = description
        self._current = 0
        self._start = time.time()

    def update(self, step: int = 1) -> None:
        self._current = min(self._current + step, self.total)
        pct = self._current / self.total * 100
        elapsed = time.time() - self._start
        eta = (elapsed / self._current) * (self.total - self._current) if self._current else 0
        print(
            f"\r{self.description}: {pct:5.1f}%  "
            f"elapsed={format_time(elapsed)}  "
            f"ETA={format_time(eta)}   ",
            end="",
            flush=True,
        )
        if self._current >= self.total:
            print()  # newline on completion

    def done(self) -> None:
        self._current = self.total
        self.update(0)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to [*lo*, *hi*]."""
    return max(lo, min(hi, value))


def batch(items: List, size: int) -> List[List]:
    """Split *items* into sub-lists of *size* items each."""
    return [items[i : i + size] for i in range(0, len(items), size)]
