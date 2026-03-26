"""
Decision Engine – segment scoring, filtering, and merging.

Takes a list of VideoSegment objects produced by the VideoProcessor,
applies scoring weights, filters low-value segments, merges adjacent
segments that are very close together, and returns the final ordered
list of segments to be rendered.
"""

from typing import List

from core.video_processor import VideoSegment
from utils.logger import get_logger

logger = get_logger(__name__)


class DecisionEngine:
    """
    Score, filter, and merge video segments.

    Parameters
    ----------
    config : object
        Application config (MIN_SEGMENT_SCORE, MIN_SEGMENT_DURATION,
        MAX_SEGMENT_DURATION, GAP_FILL_THRESHOLD).
    """

    # Scoring weights (must sum to 1.0)
    WEIGHT_AUDIO_ENERGY = 0.5
    WEIGHT_HAS_MOTION = 0.3
    WEIGHT_DURATION = 0.2

    def __init__(self, config):
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, segments: List[VideoSegment]) -> List[VideoSegment]:
        """
        Full decision pipeline.

        1. Re-score segments with weighted formula.
        2. Filter out low-score / too-short / too-long segments.
        3. Merge adjacent segments that are very close together.
        4. Return sorted list ready for rendering.
        """
        if not segments:
            logger.warning("Decision engine received an empty segment list")
            return []

        logger.info("Decision engine processing %d candidate segments", len(segments))

        scored = [self._compute_score(seg) for seg in segments]
        filtered = self._filter(scored)
        merged = self._merge_close(filtered)

        logger.info(
            "Decision engine: %d → filtered %d → merged %d segments",
            len(scored),
            len(filtered),
            len(merged),
        )
        return merged

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_score(self, seg: VideoSegment) -> VideoSegment:
        """
        Compute a weighted importance score for *seg* and return it
        (modifies in-place for efficiency).
        """
        max_dur = self.config.MAX_SEGMENT_DURATION

        # Audio energy component (already in [0, 1])
        audio_component = seg.score * self.WEIGHT_AUDIO_ENERGY

        # Motion component
        motion_component = (1.0 if seg.has_motion else 0.0) * self.WEIGHT_HAS_MOTION

        # Duration component: prefer segments closer to max_dur
        duration_ratio = min(1.0, seg.duration / max_dur) if max_dur > 0 else 0.0
        duration_component = duration_ratio * self.WEIGHT_DURATION

        seg.score = audio_component + motion_component + duration_component
        return seg

    def _filter(self, segments: List[VideoSegment]) -> List[VideoSegment]:
        """Remove segments that don't meet score or duration thresholds."""
        min_score = self.config.MIN_SEGMENT_SCORE
        min_dur = self.config.MIN_SEGMENT_DURATION
        max_dur = self.config.MAX_SEGMENT_DURATION

        kept = []
        for seg in segments:
            if seg.score < min_score:
                logger.debug(
                    "Dropping segment [%.2f–%.2f] score=%.3f (below threshold)",
                    seg.start, seg.end, seg.score,
                )
                continue
            if seg.duration < min_dur:
                logger.debug(
                    "Dropping segment [%.2f–%.2f] duration=%.2f s (too short)",
                    seg.start, seg.end, seg.duration,
                )
                continue
            if seg.duration > max_dur:
                logger.debug(
                    "Dropping segment [%.2f–%.2f] duration=%.2f s (too long)",
                    seg.start, seg.end, seg.duration,
                )
                continue
            kept.append(seg)

        return kept

    def _merge_close(self, segments: List[VideoSegment]) -> List[VideoSegment]:
        """
        Merge adjacent segments whose gap is <= GAP_FILL_THRESHOLD seconds.

        The merged segment inherits the higher score of the two.
        """
        if not segments:
            return []

        threshold = self.config.GAP_FILL_THRESHOLD
        segments = sorted(segments, key=lambda s: s.start)

        merged: List[VideoSegment] = [segments[0]]
        for seg in segments[1:]:
            prev = merged[-1]
            gap = seg.start - prev.end
            if gap <= threshold:
                # Extend previous segment
                prev.end = max(prev.end, seg.end)
                prev.score = max(prev.score, seg.score)
                prev.has_motion = prev.has_motion or seg.has_motion
                prev.tags = list(set(prev.tags + seg.tags))
            else:
                merged.append(seg)

        return merged
