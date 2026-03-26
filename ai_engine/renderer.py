"""
Video Renderer module for the Intelligent Video Editing System.

Responsibilities
----------------
* Execute an *EditPlan* produced by the Editor module.
* Compose the final video with all cuts, transitions, speed changes,
  zoom, colour correction, and audio normalisation applied.
* Export the result at the configured quality preset.

The renderer uses MoviePy as the primary composition engine and calls
FFmpeg directly for format conversion and codec optimisation.
"""

import logging
import os
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

from utils.config import Config
from utils.helpers import ensure_dir, cleanup_temp_files, format_time

logger = logging.getLogger("video_editor.renderer")


class Renderer:
    """Renders the final video from an *EditPlan*.

    Parameters
    ----------
    config:
        System-wide :class:`~utils.config.Config` instance.
    """

    def __init__(self, config: Config) -> None:
        self.cfg = config
        ensure_dir(config.temp_dir)
        ensure_dir(config.output_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        input_path: str,
        edit_plan: Dict,
        output_path: str,
        progress_callback=None,
    ) -> str:
        """Apply *edit_plan* to *input_path* and write the result to *output_path*.

        Parameters
        ----------
        input_path:
            Path to the source video file.
        edit_plan:
            Dict produced by :meth:`~ai_engine.editor.VideoEditor.create_edit_plan`.
        output_path:
            Destination path for the rendered video.
        progress_callback:
            Optional callable ``(pct: float, message: str) -> None`` for
            progress updates.

        Returns
        -------
        str
            Absolute path of the rendered output file.
        """
        logger.info("Rendering '%s' → '%s'", input_path, output_path)

        clips_to_keep = [c for c in edit_plan.get("clips", []) if c["type"] == "cut"]
        if not clips_to_keep:
            logger.warning("No clips to render; copying input as output.")
            self._copy_file(input_path, output_path)
            return os.path.abspath(output_path)

        effects: List[Dict] = edit_plan.get("effects", [])
        global_settings: Dict = edit_plan.get("global", {})

        try:
            from moviepy.editor import (  # type: ignore
                VideoFileClip,
                concatenate_videoclips,
                AudioFileClip,
                CompositeVideoClip,
            )
        except ImportError as exc:
            raise RuntimeError(
                "MoviePy is required for rendering. Install it with: "
                "pip install moviepy"
            ) from exc

        _report(progress_callback, 5, "Loading source video …")

        source = VideoFileClip(input_path)
        target_w, target_h = global_settings.get(
            "output_resolution", self.cfg.resolution
        )
        target_fps = global_settings.get("output_fps", self.cfg.fps)

        # ---- Build per-clip MoviePy clip objects ----
        processed_clips = []
        total = len(clips_to_keep)

        for i, clip_info in enumerate(clips_to_keep):
            pct = 10 + int(i / total * 50)
            _report(progress_callback, pct, f"Processing clip {i+1}/{total} …")

            try:
                mpy_clip = source.subclip(
                    clip_info["start_time"], clip_info["end_time"]
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not subclip at [%.2f, %.2f]: %s",
                               clip_info["start_time"], clip_info["end_time"], exc)
                continue

            # Apply effects that target this clip index
            clip_effects = [
                e for e in effects if e.get("clip_index") == i
            ]
            mpy_clip = self._apply_effects(mpy_clip, clip_effects)

            # Resize to target resolution
            mpy_clip = mpy_clip.resize((target_w, target_h))

            processed_clips.append(mpy_clip)

        if not processed_clips:
            source.close()
            raise RuntimeError("All clips were empty after processing.")

        _report(progress_callback, 65, "Concatenating clips …")

        # ---- Add transitions ----
        final_clip = self._concatenate_with_transitions(
            processed_clips, effects, target_fps
        )

        # ---- Global audio normalisation ----
        target_db = global_settings.get("audio_normalization_db")
        if target_db is not None and final_clip.audio is not None:
            _report(progress_callback, 75, "Normalising audio …")
            final_clip = self._normalise_audio(final_clip, target_db)

        _report(progress_callback, 80, "Writing output file …")

        # Ensure output directory exists
        ensure_dir(os.path.dirname(os.path.abspath(output_path)))

        final_clip.write_videofile(
            output_path,
            codec=self.cfg.output_codec,
            audio_codec=self.cfg.audio_codec,
            fps=target_fps,
            bitrate=self.cfg.video_bitrate,
            audio_bitrate=self.cfg.audio_bitrate,
            preset="fast",
            threads=2,
            logger=None,  # suppress moviepy verbose output
        )

        source.close()
        final_clip.close()
        for c in processed_clips:
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass

        _report(progress_callback, 100, "Rendering complete.")
        logger.info("Rendering complete: '%s'", output_path)
        return os.path.abspath(output_path)

    # ------------------------------------------------------------------
    # Effect application
    # ------------------------------------------------------------------

    def _apply_effects(self, clip, effects: List[Dict]):
        """Apply a list of effect dicts to a MoviePy clip."""
        for effect in effects:
            etype = effect.get("type")
            params = effect.get("params", {})

            if etype == "speed":
                clip = self._apply_speed(clip, params)
            elif etype == "zoom":
                clip = self._apply_zoom(clip, params)
            elif etype == "color":
                clip = self._apply_color(clip, params)
            # "transition" and "audio_norm" are handled globally

        return clip

    def _apply_speed(self, clip, params: Dict):
        factor = float(params.get("factor", 1.0))
        if abs(factor - 1.0) < 0.01:
            return clip
        return clip.speedx(factor)

    def _apply_zoom(self, clip, params: Dict):
        zoom_factor = float(params.get("zoom_factor", 1.0))
        if zoom_factor <= 1.0:
            return clip
        w, h = clip.size
        roi = params.get("roi")  # (x, y, rw, rh) in original coords

        if roi:
            rx, ry, rw, rh = roi
            # Centre of ROI
            cx = rx + rw // 2
            cy = ry + rh // 2
        else:
            cx, cy = w // 2, h // 2

        # Crop to a zoomed region centred on (cx, cy), then resize back
        new_w = int(w / zoom_factor)
        new_h = int(h / zoom_factor)
        x1 = max(0, cx - new_w // 2)
        y1 = max(0, cy - new_h // 2)
        x2 = min(w, x1 + new_w)
        y2 = min(h, y1 + new_h)

        try:
            return clip.crop(x1=x1, y1=y1, x2=x2, y2=y2).resize((w, h))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Zoom crop failed: %s", exc)
            return clip

    def _apply_color(self, clip, params: Dict):
        """Apply brightness/contrast correction using MoviePy's fl_image."""
        strength = float(params.get("strength", 0.5))
        target_brightness = float(params.get("target_brightness", 128))

        if strength < 0.01:
            return clip

        try:
            import numpy as np

            def correct_frame(frame: np.ndarray) -> np.ndarray:
                mean = float(np.mean(frame))
                if mean < 1:
                    return frame
                scale = 1.0 + strength * (target_brightness / mean - 1.0)
                corrected = np.clip(frame * scale, 0, 255).astype(np.uint8)
                return corrected

            return clip.fl_image(correct_frame)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Color correction failed: %s", exc)
            return clip

    # ------------------------------------------------------------------
    # Concatenation with transitions
    # ------------------------------------------------------------------

    def _concatenate_with_transitions(
        self,
        clips: list,
        effects: List[Dict],
        fps: int,
    ):
        """Concatenate clips, inserting cross-fade transitions where planned."""
        from moviepy.editor import concatenate_videoclips  # type: ignore

        # Collect transition durations per clip index
        trans_by_idx: Dict[int, float] = {}
        for e in effects:
            if e.get("type") == "transition":
                idx = e.get("clip_index", -1)
                dur = float(e.get("params", {}).get("duration", 0.5))
                trans_by_idx[idx] = dur

        if not trans_by_idx:
            return concatenate_videoclips(clips, method="compose")

        # Apply cross-fade between consecutive clips
        result_clips = [clips[0]]
        for i in range(1, len(clips)):
            prev = result_clips[-1]
            curr = clips[i]
            trans_dur = trans_by_idx.get(i - 1, 0.0)

            if trans_dur > 0 and prev.duration > trans_dur and curr.duration > trans_dur:
                try:
                    curr = curr.crossfadein(trans_dur)
                    result_clips[-1] = prev.crossfadeout(trans_dur)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Crossfade failed between clips %d and %d: %s", i - 1, i, exc)

            result_clips.append(curr)

        return concatenate_videoclips(result_clips, method="compose")

    # ------------------------------------------------------------------
    # Audio normalisation
    # ------------------------------------------------------------------

    def _normalise_audio(self, clip, target_db: float):
        """Scale the clip's audio so its RMS level equals *target_db* dBFS."""
        try:
            import numpy as np

            audio = clip.audio
            if audio is None:
                return clip

            frames = audio.to_soundarray(fps=audio.fps)
            rms = float(np.sqrt(np.mean(frames ** 2)))
            if rms < 1e-9:
                return clip

            # target_db is negative dBFS (e.g. -16)
            target_rms = 10 ** (target_db / 20.0)
            gain = target_rms / rms

            def scale_audio(get_frame, t):
                frame = get_frame(t)
                return np.clip(frame * gain, -1.0, 1.0)

            new_audio = audio.fl(scale_audio)
            return clip.set_audio(new_audio)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Audio normalisation failed: %s", exc)
            return clip

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _copy_file(src: str, dst: str) -> None:
        import shutil
        ensure_dir(os.path.dirname(os.path.abspath(dst)))
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _report(callback, pct: float, message: str) -> None:
    logger.info("[%3.0f%%] %s", pct, message)
    if callback:
        try:
            callback(pct, message)
        except Exception:  # noqa: BLE001
            pass
