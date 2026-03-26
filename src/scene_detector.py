"""
Scene Detection module.
Detects scene changes, identifies key frames, and calculates shot boundaries
using histogram analysis and frame differencing.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import Config
from .utils import compute_histogram, frame_to_timestamp, histogram_difference

logger = logging.getLogger("video_editing_system.scene_detector")


@dataclass
class Scene:
    """Represents a detected scene (shot)."""

    start_frame: int
    end_frame: int
    fps: float
    key_frame: int = -1
    score: float = 0.0
    histogram: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def start_time(self) -> float:
        return frame_to_timestamp(self.start_frame, self.fps)

    @property
    def end_time(self) -> float:
        return frame_to_timestamp(self.end_frame, self.fps)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame

    def __str__(self) -> str:
        return (
            f"Scene(frames {self.start_frame}-{self.end_frame}, "
            f"{self.start_time:.2f}s-{self.end_time:.2f}s, "
            f"key_frame={self.key_frame})"
        )


class SceneDetector:
    """
    Detect scene changes and transitions in a video.

    Uses histogram-based frame differencing to find shot boundaries,
    then selects key frames that best represent each scene.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._scores: List[float] = []

    @property
    def scores(self) -> List[float]:
        """Per-frame scene-change scores computed during the last detection run."""
        return list(self._scores)

    def detect(
        self,
        frames: List[Tuple[int, np.ndarray]],
        fps: float = 25.0,
    ) -> List[Scene]:
        """
        Detect scenes from a sequence of (frame_number, frame) tuples.

        Parameters
        ----------
        frames : list of (int, ndarray)
            Sequence of frame-index/frame pairs as produced by
            ``VideoProcessor.frames()``.
        fps : float
            Video frame rate used to compute timestamps.

        Returns
        -------
        list[Scene]
            Detected scenes sorted by start frame.
        """
        if not frames:
            return []

        threshold = self.config.scene_threshold
        min_len = self.config.min_scene_length
        bins = self.config.histogram_bins

        # Compute per-frame histogram differences
        histograms = [compute_histogram(f, bins) for _, f in frames]
        frame_numbers = [idx for idx, _ in frames]

        differences: List[float] = [0.0]
        for i in range(1, len(histograms)):
            diff = histogram_difference(histograms[i - 1], histograms[i])
            differences.append(diff)

        # Normalise differences to [0, 1]
        max_diff = max(differences) if differences else 1.0
        if max_diff > 0:
            norm_diffs = [d / max_diff for d in differences]
        else:
            norm_diffs = differences

        self._scores = norm_diffs

        # Identify cut points
        cut_indices: List[int] = []
        for i, score in enumerate(norm_diffs):
            if score >= threshold:
                # Suppress cuts that are too close together
                if not cut_indices or (i - cut_indices[-1]) >= min_len:
                    cut_indices.append(i)

        # Build Scene objects from cut points
        boundaries = [0] + cut_indices + [len(frames)]
        scenes: List[Scene] = []

        for b in range(len(boundaries) - 1):
            start_pos = boundaries[b]
            end_pos = boundaries[b + 1]

            if (end_pos - start_pos) < max(1, min_len // 2):
                continue

            start_fn = frame_numbers[start_pos]
            end_fn = frame_numbers[end_pos - 1]

            key_frame_pos = self._select_key_frame(
                frames[start_pos:end_pos], histograms[start_pos:end_pos]
            )
            key_fn = frame_numbers[start_pos + key_frame_pos]
            scene_hist = histograms[start_pos + key_frame_pos]
            scene_score = norm_diffs[start_pos] if start_pos < len(norm_diffs) else 0.0

            scenes.append(
                Scene(
                    start_frame=start_fn,
                    end_frame=end_fn,
                    fps=fps,
                    key_frame=key_fn,
                    score=scene_score,
                    histogram=scene_hist,
                )
            )

        logger.info(
            "Detected %d scenes from %d frames (threshold=%.2f)",
            len(scenes),
            len(frames),
            threshold,
        )
        return scenes

    def _select_key_frame(
        self,
        frames: List[Tuple[int, np.ndarray]],
        histograms: List[np.ndarray],
    ) -> int:
        """
        Select the index (within the scene's local frame list) of the key frame.

        Strategy: choose the frame whose histogram is closest to the scene's
        mean histogram (most representative frame).
        """
        if not frames:
            return 0
        if len(frames) == 1:
            return 0

        mean_hist = np.mean(histograms, axis=0)
        distances = [histogram_difference(h, mean_hist) for h in histograms]
        return int(np.argmin(distances))

    def detect_transitions(
        self, frames: List[Tuple[int, np.ndarray]], fps: float = 25.0
    ) -> List[Tuple[int, float]]:
        """
        Return a list of (frame_number, score) pairs marking transition frames.

        Lighter-weight than ``detect()`` — returns raw cut points without
        grouping them into Scene objects.
        """
        if not frames:
            return []

        bins = self.config.histogram_bins
        threshold = self.config.scene_threshold

        histograms = [compute_histogram(f, bins) for _, f in frames]
        frame_numbers = [idx for idx, _ in frames]

        transitions: List[Tuple[int, float]] = []
        for i in range(1, len(histograms)):
            diff = histogram_difference(histograms[i - 1], histograms[i])
            if diff >= threshold:
                transitions.append((frame_numbers[i], diff))

        return transitions
