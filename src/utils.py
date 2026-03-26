"""
Utility functions for the Intelligent Video Editing System.
"""

import logging
import os
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Configure and return a logger for the system."""
    logger = logging.getLogger("video_editing_system")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True) if os.path.dirname(log_file) else None
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist and return the path."""
    os.makedirs(path, exist_ok=True)
    return path


def frame_to_timestamp(frame_number: int, fps: float) -> float:
    """Convert a frame number to a timestamp in seconds."""
    if fps <= 0:
        raise ValueError(f"FPS must be positive, got {fps}")
    return frame_number / fps


def timestamp_to_frame(timestamp: float, fps: float) -> int:
    """Convert a timestamp in seconds to the nearest frame number."""
    if fps <= 0:
        raise ValueError(f"FPS must be positive, got {fps}")
    return int(round(timestamp * fps))


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def resize_frame(frame: np.ndarray, width: int = 640, height: Optional[int] = None) -> np.ndarray:
    """Resize a frame while maintaining aspect ratio."""
    h, w = frame.shape[:2]
    if height is None:
        ratio = width / w
        height = int(h * ratio)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def compute_histogram(frame: np.ndarray, bins: int = 256) -> np.ndarray:
    """Compute a normalised grayscale histogram for a frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def histogram_difference(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Compute the chi-squared distance between two histograms."""
    return float(cv2.compareHist(
        hist1.reshape(-1, 1).astype(np.float32),
        hist2.reshape(-1, 1).astype(np.float32),
        cv2.HISTCMP_CHISQR,
    ))


def draw_bounding_box(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    label: str = "",
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw a labelled bounding box on a frame."""
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    if label:
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x, y - text_h - 4), (x + text_w, y), color, -1)
        cv2.putText(frame, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame


def save_frame(frame: np.ndarray, path: str) -> bool:
    """Save a single frame as an image file."""
    try:
        ensure_dir(os.path.dirname(path)) if os.path.dirname(path) else None
        return cv2.imwrite(path, frame)
    except Exception:
        return False


class Timer:
    """Simple context-manager timer."""

    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed = time.perf_counter() - self._start

    def __str__(self) -> str:
        label = f"{self.name}: " if self.name else ""
        return f"{label}{self.elapsed:.3f}s"
