"""
Auto-Editing Engine.
Combines scene detection, object detection, and motion analysis to generate
automatic cut suggestions, highlight reels, and video summaries.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import Config, EditingProfile
from .motion_analyzer import MotionAnalyzer, MotionResult, MotionSegment
from .object_detector import Detection, ObjectDetector, Track
from .scene_detector import Scene, SceneDetector
from .utils import frame_to_timestamp

logger = logging.getLogger("video_editing_system.auto_editor")


@dataclass
class EditSuggestion:
    """A single editing suggestion produced by the Auto-Editing Engine."""

    start_frame: int
    end_frame: int
    fps: float
    suggestion_type: str  # "cut", "highlight", "summary_segment", "transition"
    score: float = 0.0
    reason: str = ""
    metadata: dict = field(default_factory=dict)

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
        return (
            f"EditSuggestion({self.suggestion_type}, "
            f"frames {self.start_frame}-{self.end_frame}, "
            f"{self.start_time:.2f}s-{self.end_time:.2f}s, "
            f"score={self.score:.2f}, reason='{self.reason}')"
        )


@dataclass
class EditingPlan:
    """Complete editing plan produced by the AutoEditor."""

    suggestions: List[EditSuggestion] = field(default_factory=list)
    highlights: List[EditSuggestion] = field(default_factory=list)
    summary_segments: List[EditSuggestion] = field(default_factory=list)
    total_frames: int = 0
    fps: float = 25.0

    @property
    def total_duration(self) -> float:
        return frame_to_timestamp(self.total_frames, self.fps)

    @property
    def summary_duration(self) -> float:
        return sum(s.duration for s in self.summary_segments)

    def __str__(self) -> str:
        return (
            f"EditingPlan("
            f"{len(self.suggestions)} suggestions, "
            f"{len(self.highlights)} highlights, "
            f"{len(self.summary_segments)} summary segments, "
            f"total={self.total_duration:.1f}s, "
            f"summary={self.summary_duration:.1f}s)"
        )


class AutoEditor:
    """
    Automatic video editing engine.

    Orchestrates scene detection, motion analysis, and object detection
    to produce actionable editing plans.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.scene_detector = SceneDetector(self.config)
        self.motion_analyzer = MotionAnalyzer(self.config)
        self.object_detector = ObjectDetector(self.config)

    def generate_plan(
        self,
        frames: List[Tuple[int, np.ndarray]],
        fps: float = 25.0,
    ) -> EditingPlan:
        """
        Generate a complete editing plan from a sequence of frames.

        Parameters
        ----------
        frames : list of (int, ndarray)
            Video frames as (frame_number, ndarray) tuples.
        fps : float
            Video frame rate.

        Returns
        -------
        EditingPlan
            All suggestions, highlights, and summary segments.
        """
        if not frames:
            return EditingPlan(fps=fps)

        logger.info(
            "Generating editing plan for %d frames at %.2f fps", len(frames), fps
        )

        total_frames = frames[-1][0] + 1

        # 1. Scene detection
        scenes = self.scene_detector.detect(frames, fps)

        # 2. Motion analysis
        motion_results = self.motion_analyzer.analyze(frames, fps)
        motion_segments = self.motion_analyzer.segment(motion_results, fps)

        # 3. Object detection (sub-sampled for efficiency)
        sample_step = max(1, len(frames) // 50)
        sample_frames = frames[::sample_step]
        frame_detections = self.object_detector.detect_sequence(sample_frames)
        tracks = self.object_detector.track_objects(frame_detections)

        # 4. Build suggestions
        cut_suggestions = self._build_cut_suggestions(scenes, fps)
        highlights = self._build_highlights(motion_segments, fps)
        summary_segs = self._build_summary(
            scenes, motion_segments, tracks, fps, total_frames
        )

        plan = EditingPlan(
            suggestions=cut_suggestions,
            highlights=highlights,
            summary_segments=summary_segs,
            total_frames=total_frames,
            fps=fps,
        )
        logger.info("Editing plan ready: %s", plan)
        return plan

    def _build_cut_suggestions(
        self, scenes: List[Scene], fps: float
    ) -> List[EditSuggestion]:
        """Create cut suggestions from scene boundaries."""
        suggestions: List[EditSuggestion] = []
        for scene in scenes:
            suggestions.append(
                EditSuggestion(
                    start_frame=scene.start_frame,
                    end_frame=scene.end_frame,
                    fps=fps,
                    suggestion_type="cut",
                    score=scene.score,
                    reason="Scene boundary detected",
                    metadata={"key_frame": scene.key_frame},
                )
            )
        return suggestions

    def _build_highlights(
        self, motion_segments: List[MotionSegment], fps: float
    ) -> List[EditSuggestion]:
        """Create highlight suggestions from high-motion segments."""
        highlights: List[EditSuggestion] = []
        for seg in motion_segments:
            if seg.is_highlight:
                highlights.append(
                    EditSuggestion(
                        start_frame=seg.start_frame,
                        end_frame=seg.end_frame,
                        fps=fps,
                        suggestion_type="highlight",
                        score=seg.mean_intensity,
                        reason=f"High motion segment (intensity={seg.mean_intensity:.2f})",
                    )
                )
        highlights.sort(key=lambda s: s.score, reverse=True)
        return highlights

    def _build_summary(
        self,
        scenes: List[Scene],
        motion_segments: List[MotionSegment],
        tracks: List[Track],
        fps: float,
        total_frames: int,
    ) -> List[EditSuggestion]:
        """
        Select the most representative scenes to form a concise summary.

        Strategy:
        - Score each scene by motion intensity overlap and object presence.
        - Select scenes whose cumulative duration is within ``summary_ratio``.
        """
        if not scenes:
            return []

        target_duration = total_frames / fps * self.config.summary_ratio

        # Map motion intensity onto scenes
        motion_by_frame: Dict[int, float] = {}
        for seg in motion_segments:
            for fn in range(seg.start_frame, seg.end_frame + 1):
                motion_by_frame[fn] = seg.mean_intensity

        # Frames that contain detections
        detection_frames = set()
        for track in tracks:
            for det in track.detections:
                detection_frames.add(det.frame_number)

        scored_scenes: List[Tuple[float, Scene]] = []
        for scene in scenes:
            scene_frames = range(scene.start_frame, scene.end_frame + 1)
            motion_vals = [motion_by_frame.get(f, 0.0) for f in scene_frames]
            mean_motion = float(np.mean(motion_vals)) if motion_vals else 0.0
            detection_overlap = sum(1 for f in scene_frames if f in detection_frames)
            det_score = min(1.0, detection_overlap / max(1, scene.frame_count) * 5)
            combined = 0.6 * mean_motion + 0.4 * det_score
            scored_scenes.append((combined, scene))

        scored_scenes.sort(key=lambda x: x[0], reverse=True)

        summary: List[EditSuggestion] = []
        accumulated = 0.0
        for score, scene in scored_scenes:
            if accumulated >= target_duration:
                break
            summary.append(
                EditSuggestion(
                    start_frame=scene.start_frame,
                    end_frame=scene.end_frame,
                    fps=fps,
                    suggestion_type="summary_segment",
                    score=score,
                    reason="Selected for summary (high relevance)",
                    metadata={"key_frame": scene.key_frame},
                )
            )
            accumulated += scene.duration

        # Sort by start frame for chronological order
        summary.sort(key=lambda s: s.start_frame)
        return summary

    def detect_transitions(
        self, frames: List[Tuple[int, np.ndarray]], fps: float = 25.0
    ) -> List[EditSuggestion]:
        """
        Detect visual transitions and return them as EditSuggestion objects.

        Parameters
        ----------
        frames : list of (int, ndarray)
            Video frames.
        fps : float
            Frame rate.

        Returns
        -------
        list[EditSuggestion]
        """
        transitions = self.scene_detector.detect_transitions(frames, fps)
        return [
            EditSuggestion(
                start_frame=fn,
                end_frame=fn,
                fps=fps,
                suggestion_type="transition",
                score=score,
                reason=f"Visual transition detected (score={score:.2f})",
            )
            for fn, score in transitions
        ]

    def close(self) -> None:
        """Release resources held by sub-components."""
        self.object_detector.close()

    def __enter__(self) -> "AutoEditor":
        return self

    def __exit__(self, *_) -> None:
        self.close()
