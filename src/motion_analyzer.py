"""
Motion Analysis module.
Analyses motion patterns, detects static vs. dynamic scenes, and suggests
optimal cut points based on motion intensity.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import Config
from .utils import frame_to_timestamp

logger = logging.getLogger("video_editing_system.motion_analyzer")


@dataclass
class MotionResult:
    """Motion analysis result for a single frame."""

    frame_number: int
    intensity: float  # normalised [0, 1]
    is_dynamic: bool
    flow_magnitude: float = 0.0
    dominant_direction: Optional[Tuple[float, float]] = None

    def __str__(self) -> str:
        state = "dynamic" if self.is_dynamic else "static"
        return (
            f"MotionResult(frame={self.frame_number}, "
            f"intensity={self.intensity:.3f}, {state})"
        )


@dataclass
class MotionSegment:
    """A contiguous segment of frames sharing similar motion characteristics."""

    start_frame: int
    end_frame: int
    fps: float
    mean_intensity: float
    is_highlight: bool = False
    label: str = ""

    @property
    def start_time(self) -> float:
        return frame_to_timestamp(self.start_frame, self.fps)

    @property
    def end_time(self) -> float:
        return frame_to_timestamp(self.end_frame, self.fps)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def __str__(self) -> str:
        tag = " [HIGHLIGHT]" if self.is_highlight else ""
        return (
            f"MotionSegment({self.label}{tag}, "
            f"frames {self.start_frame}-{self.end_frame}, "
            f"intensity={self.mean_intensity:.3f})"
        )


class MotionAnalyzer:
    """
    Analyse motion patterns in a video sequence.

    Uses dense optical flow (Farneback) to compute per-frame motion intensity,
    classify frames as static/dynamic, and identify highlight segments.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

    def analyze(
        self,
        frames: List[Tuple[int, np.ndarray]],
        fps: float = 25.0,
    ) -> List[MotionResult]:
        """
        Compute per-frame motion intensity using optical flow.

        Parameters
        ----------
        frames : list of (int, ndarray)
            Sequence of (frame_number, frame) pairs.
        fps : float
            Video frame rate (used for logging only here).

        Returns
        -------
        list[MotionResult]
            One result per input frame.
        """
        if len(frames) < 2:
            return [
                MotionResult(
                    frame_number=fn, intensity=0.0, is_dynamic=False
                )
                for fn, _ in frames
            ]

        threshold = self.config.motion_threshold
        bk = self.config.motion_blur_kernel
        blur_k = max(3, bk if bk % 2 == 1 else bk + 1)  # must be odd

        results: List[MotionResult] = []
        magnitudes: List[float] = []

        # Convert to grayscale
        gray_frames = []
        for fn, frame in frames:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.copy()
            gray = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
            gray_frames.append((fn, gray))

        # Compute optical flow between consecutive frames
        raw_mags: List[float] = [0.0]
        for i in range(1, len(gray_frames)):
            prev_gray = gray_frames[i - 1][1]
            curr_gray = gray_frames[i][1]
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                curr_gray,
                None,
                pyr_scale=0.5,
                levels=self.config.optical_flow_levels,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mean_mag = float(np.mean(mag))
            raw_mags.append(mean_mag)

        # Normalise magnitudes to [0, 1]
        max_mag = max(raw_mags) if raw_mags else 1.0
        norm_mags = [m / max_mag if max_mag > 0 else 0.0 for m in raw_mags]

        for i, (fn, _) in enumerate(frames):
            intensity = norm_mags[i]
            is_dynamic = intensity >= threshold

            # Dominant direction (only for dynamic frames)
            direction = None
            if is_dynamic and i > 0:
                prev_gray = gray_frames[i - 1][1]
                curr_gray = gray_frames[i][1]
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, curr_gray, None,
                    pyr_scale=0.5, levels=2, winsize=15,
                    iterations=2, poly_n=5, poly_sigma=1.1, flags=0,
                )
                mean_vx = float(np.mean(flow[..., 0]))
                mean_vy = float(np.mean(flow[..., 1]))
                direction = (mean_vx, mean_vy)

            results.append(
                MotionResult(
                    frame_number=fn,
                    intensity=intensity,
                    is_dynamic=is_dynamic,
                    flow_magnitude=raw_mags[i],
                    dominant_direction=direction,
                )
            )

        logger.info(
            "Motion analysis: %d frames, %.1f%% dynamic",
            len(results),
            100 * sum(r.is_dynamic for r in results) / max(len(results), 1),
        )
        return results

    def segment(
        self,
        results: List[MotionResult],
        fps: float = 25.0,
        min_segment_frames: int = 5,
    ) -> List[MotionSegment]:
        """
        Group MotionResult objects into contiguous MotionSegment regions.

        Parameters
        ----------
        results : list[MotionResult]
            Per-frame motion results from ``analyze()``.
        fps : float
            Video frame rate.
        min_segment_frames : int
            Minimum number of frames to form a segment.

        Returns
        -------
        list[MotionSegment]
            Detected motion segments.
        """
        if not results:
            return []

        highlight_threshold = self.config.highlight_motion_threshold
        segments: List[MotionSegment] = []
        current_is_dynamic = results[0].is_dynamic
        seg_start = 0

        def _flush(start_idx: int, end_idx: int, is_dynamic: bool) -> None:
            if (end_idx - start_idx) < min_segment_frames:
                return
            intensities = [results[j].intensity for j in range(start_idx, end_idx)]
            mean_int = float(np.mean(intensities))
            seg = MotionSegment(
                start_frame=results[start_idx].frame_number,
                end_frame=results[end_idx - 1].frame_number,
                fps=fps,
                mean_intensity=mean_int,
                is_highlight=mean_int >= highlight_threshold,
                label="dynamic" if is_dynamic else "static",
            )
            segments.append(seg)

        for i in range(1, len(results)):
            if results[i].is_dynamic != current_is_dynamic:
                _flush(seg_start, i, current_is_dynamic)
                seg_start = i
                current_is_dynamic = results[i].is_dynamic

        _flush(seg_start, len(results), current_is_dynamic)
        return segments

    def get_highlight_frames(
        self, results: List[MotionResult], top_n: int = 10
    ) -> List[int]:
        """Return the top-N frame numbers with the highest motion intensity."""
        sorted_results = sorted(results, key=lambda r: r.intensity, reverse=True)
        return [r.frame_number for r in sorted_results[:top_n]]

    def suggest_cuts(
        self, segments: List[MotionSegment], fps: float = 25.0
    ) -> List[Tuple[int, str]]:
        """
        Suggest cut points based on motion segment transitions.

        Returns
        -------
        list of (frame_number, reason)
        """
        suggestions: List[Tuple[int, str]] = []
        for i in range(1, len(segments)):
            prev = segments[i - 1]
            curr = segments[i]
            if prev.label == "dynamic" and curr.label == "static":
                suggestions.append(
                    (curr.start_frame, "Transition from dynamic to static scene")
                )
            elif prev.label == "static" and curr.label == "dynamic":
                suggestions.append(
                    (curr.start_frame, "Transition from static to dynamic scene")
                )
            elif curr.is_highlight and not prev.is_highlight:
                suggestions.append(
                    (curr.start_frame, "Highlight segment begins")
                )
        return suggestions
