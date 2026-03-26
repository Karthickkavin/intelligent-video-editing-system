"""
Video Editor module for the Intelligent Video Editing System.

Responsibilities
----------------
* Convert an *AnalysisResult* (from Analyzer) into an *EditPlan* dict.
* Each edit decision is a plain dict that the Renderer can execute.

Edit types
----------
cut          – Keep a time range as-is.
transition   – Insert a crossfade / dissolve between two clips.
speed        – Change playback speed of a clip.
zoom         – Apply a progressive zoom into a region of a clip.
color        – Apply colour correction to a clip.
audio_norm   – Normalise the audio level of a clip.
remove       – Drop a time range (e.g. silence).

EditPlan structure
------------------
{
    "clips": [
        {
            "type":        "cut" | "remove",
            "start_time":  float,
            "end_time":    float,
        },
        ...
    ],
    "effects": [
        {
            "type":        "transition" | "speed" | "zoom" | "color" | "audio_norm",
            "clip_index":  int,   # 0-based index into clips list
            "params":      dict,  # type-specific parameters
        },
        ...
    ],
    "global": {
        "audio_normalization_db": float | None,
        "output_resolution":     (int, int),
        "output_fps":            int,
    }
}
"""

import logging
import random
from typing import Dict, List, Optional, Tuple

from utils.config import Config
from utils.helpers import clamp

logger = logging.getLogger("video_editor.editor")


# ---------------------------------------------------------------------------
# Transition types available to the renderer
# ---------------------------------------------------------------------------
TRANSITION_TYPES = ["fade", "crossfade", "dissolve"]


class VideoEditor:
    """Creates an *EditPlan* from an *AnalysisResult*.

    Parameters
    ----------
    config:
        System-wide :class:`~utils.config.Config` instance.
    """

    def __init__(self, config: Config) -> None:
        self.cfg = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_edit_plan(self, analysis: Dict) -> Dict:
        """Build a complete edit plan from *analysis*.

        Parameters
        ----------
        analysis:
            Dict returned by :meth:`~ai_engine.analyzer.Analyzer.analyze`.

        Returns
        -------
        dict
            An *EditPlan* dict (see module docstring for schema).
        """
        logger.info("Creating edit plan …")

        video_info = analysis["video_info"]
        scenes: List[Dict] = analysis.get("scenes", [])
        audio_segments: List[Dict] = analysis.get("audio_segments", [])
        faces_by_scene: Dict = analysis.get("faces_by_scene", {})
        objects_by_scene: Dict = analysis.get("objects_by_scene", {})

        # --- Build clip list from scenes, removing silences ---
        clips = self._build_clips(scenes, audio_segments)

        # --- Build effects list ---
        effects: List[Dict] = []

        if self.cfg.enable_transitions:
            effects.extend(self._plan_transitions(clips))

        if self.cfg.enable_speed_adjustment:
            effects.extend(self._plan_speed_adjustments(clips, analysis.get("motion_scores", [])))

        if self.cfg.enable_auto_zoom:
            effects.extend(
                self._plan_auto_zoom(clips, faces_by_scene, objects_by_scene)
            )

        if self.cfg.enable_color_correction:
            effects.extend(
                self._plan_color_correction(clips, analysis.get("brightness_scores", []))
            )

        global_settings: Dict = {
            "audio_normalization_db": (
                self.cfg.audio_target_db if self.cfg.enable_audio_normalization else None
            ),
            "output_resolution": self.cfg.resolution,
            "output_fps": self.cfg.fps,
        }

        plan = {
            "clips": clips,
            "effects": effects,
            "global": global_settings,
        }

        kept = sum(1 for c in clips if c["type"] == "cut")
        removed = sum(1 for c in clips if c["type"] == "remove")
        logger.info(
            "Edit plan: %d clip(s) kept, %d removed, %d effect(s).",
            kept,
            removed,
            len(effects),
        )
        return plan

    # ------------------------------------------------------------------
    # Clip building
    # ------------------------------------------------------------------

    def _build_clips(
        self,
        scenes: List[Dict],
        audio_segments: List[Dict],
    ) -> List[Dict]:
        """Combine scene list and silence removal into a flat clip list."""
        if not scenes:
            return []

        clips: List[Dict] = []

        for scene in scenes:
            # Check if this scene overlaps with a removable silence segment
            scene_clips = self._split_scene_by_silence(scene, audio_segments)
            clips.extend(scene_clips)

        return clips

    def _split_scene_by_silence(
        self, scene: Dict, audio_segments: List[Dict]
    ) -> List[Dict]:
        """Split a single scene into keep/remove clips based on silence segments."""
        start = scene["start_time"]
        end = scene["end_time"]
        clips: List[Dict] = []

        if not self.cfg.enable_silence_removal:
            clips.append({"type": "cut", "start_time": start, "end_time": end})
            return clips

        # Find silence segments that fall within this scene
        cursor = start
        for seg in audio_segments:
            if seg["end_time"] <= start or seg["start_time"] >= end:
                continue  # outside scene

            seg_start = max(seg["start_time"], start)
            seg_end = min(seg["end_time"], end)

            if cursor < seg_start:
                clips.append({"type": "cut", "start_time": cursor, "end_time": seg_start})

            if seg["is_silence"] and (seg_end - seg_start) >= self.cfg.min_silence_duration:
                clips.append({"type": "remove", "start_time": seg_start, "end_time": seg_end})
            else:
                clips.append({"type": "cut", "start_time": seg_start, "end_time": seg_end})

            cursor = seg_end

        if cursor < end:
            clips.append({"type": "cut", "start_time": cursor, "end_time": end})

        return clips if clips else [{"type": "cut", "start_time": start, "end_time": end}]

    # ------------------------------------------------------------------
    # Transition planning
    # ------------------------------------------------------------------

    def _plan_transitions(self, clips: List[Dict]) -> List[Dict]:
        """Insert transitions between consecutive *cut* clips."""
        effects: List[Dict] = []
        kept_indices = [i for i, c in enumerate(clips) if c["type"] == "cut"]

        for i in range(len(kept_indices) - 1):
            idx = kept_indices[i]
            effects.append(
                {
                    "type": "transition",
                    "clip_index": idx,
                    "params": {
                        "transition_type": random.choice(TRANSITION_TYPES),
                        "duration": self.cfg.transition_duration,
                    },
                }
            )

        return effects

    # ------------------------------------------------------------------
    # Speed adjustments
    # ------------------------------------------------------------------

    def _plan_speed_adjustments(
        self, clips: List[Dict], motion_scores: List[float]
    ) -> List[Dict]:
        """Vary playback speed based on motion intensity."""
        if not motion_scores:
            return []

        effects: List[Dict] = []
        lo, hi = self.cfg.speed_range  # type: ignore[misc]

        for i, clip in enumerate(clips):
            if clip["type"] != "cut":
                continue

            # Estimate clip's average motion (rough – uses overall score list)
            # Real per-clip mapping is refined in the renderer
            total = len(motion_scores)
            avg_motion = float(sum(motion_scores) / total) if total else 0.0

            # High motion → slight slow-down for drama; low motion → speed up
            if avg_motion > 0.1:  # fast action
                speed = clamp(1.0 - (avg_motion - 0.1) * 0.5, lo, 1.0)
            else:  # slow / static
                speed = clamp(1.0 + (0.1 - avg_motion) * 2.0, 1.0, hi)

            if abs(speed - 1.0) > 0.05:  # only add effect if change is meaningful
                effects.append(
                    {
                        "type": "speed",
                        "clip_index": i,
                        "params": {"factor": round(speed, 3)},
                    }
                )

        return effects

    # ------------------------------------------------------------------
    # Auto-zoom
    # ------------------------------------------------------------------

    def _plan_auto_zoom(
        self,
        clips: List[Dict],
        faces_by_scene: Dict,
        objects_by_scene: Dict,
    ) -> List[Dict]:
        """Add zoom effects to clips that contain detected faces/objects."""
        effects: List[Dict] = []
        max_zoom = self.cfg.max_zoom

        for i, clip in enumerate(clips):
            if clip["type"] != "cut":
                continue

            faces = faces_by_scene.get(i, [])
            objects = objects_by_scene.get(i, [])

            roi = self._compute_roi(faces, objects)
            if roi is None:
                continue

            x, y, w, h = roi
            effects.append(
                {
                    "type": "zoom",
                    "clip_index": i,
                    "params": {
                        "roi": (x, y, w, h),
                        "zoom_factor": max_zoom,
                        "smooth": True,
                    },
                }
            )

        return effects

    @staticmethod
    def _compute_roi(
        faces: List[Tuple],
        objects: List[Dict],
    ) -> Optional[Tuple[int, int, int, int]]:
        """Return a combined region of interest bounding box, or None."""
        boxes: List[Tuple[int, int, int, int]] = []

        for f in faces:
            if len(f) == 4:
                boxes.append(tuple(int(v) for v in f))  # type: ignore[arg-type]

        for obj in objects:
            if "bbox" in obj:
                boxes.append(tuple(int(v) for v in obj["bbox"]))  # type: ignore[arg-type]

        if not boxes:
            return None

        # Union bounding box
        xs = [b[0] for b in boxes]
        ys = [b[1] for b in boxes]
        x2s = [b[0] + b[2] for b in boxes]
        y2s = [b[1] + b[3] for b in boxes]

        x, y = min(xs), min(ys)
        x2, y2 = max(x2s), max(y2s)
        return x, y, x2 - x, y2 - y

    # ------------------------------------------------------------------
    # Colour correction
    # ------------------------------------------------------------------

    def _plan_color_correction(
        self, clips: List[Dict], brightness_scores: List[float]
    ) -> List[Dict]:
        """Add colour-correction effects to clips that need it."""
        effects: List[Dict] = []
        if not brightness_scores:
            return effects

        overall_mean = float(sum(brightness_scores) / len(brightness_scores))
        strength = self.cfg.color_correction_strength

        for i, clip in enumerate(clips):
            if clip["type"] != "cut":
                continue

            # Simple brightness/contrast normalisation target
            effects.append(
                {
                    "type": "color",
                    "clip_index": i,
                    "params": {
                        "target_brightness": clamp(overall_mean, 80, 180),
                        "strength": strength,
                        "auto_contrast": True,
                    },
                }
            )

        return effects
