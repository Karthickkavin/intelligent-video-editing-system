"""
Video Analyzer module for the Intelligent Video Editing System.

Responsibilities
----------------
* Scene / shot detection using optical flow and frame differencing
* Audio analysis: silence detection, speech vs. music classification
* Face detection and tracking
* Optional object detection
* Colour and brightness analysis
* Motion intensity estimation

All results are returned as plain Python dicts so they can be serialised,
logged, and consumed by the Editor module without tight coupling.
"""

import logging
import os
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.config import Config
from utils.helpers import format_time, get_video_info
from ai_engine.models import ModelManager

logger = logging.getLogger("video_editor.analyzer")


# ---------------------------------------------------------------------------
# Data structures (plain dicts with documented keys)
# ---------------------------------------------------------------------------

# Scene dict keys:
#   start_frame  int   – first frame index of the scene
#   end_frame    int   – last  frame index of the scene (exclusive)
#   start_time   float – start timestamp in seconds
#   end_time     float – end   timestamp in seconds
#   duration     float – duration in seconds
#   avg_motion   float – average optical-flow magnitude in the scene
#   avg_brightness float – mean pixel brightness (0-255)

# AudioSegment dict keys:
#   start_time   float
#   end_time     float
#   is_silence   bool
#   rms_db       float – RMS level in dBFS

# AnalysisResult dict keys:
#   video_info   dict
#   scenes       List[scene dict]
#   audio_segments List[AudioSegment dict]
#   faces_by_scene   {scene_index: List[(x,y,w,h)]}
#   objects_by_scene {scene_index: List[detection dict]}
#   motion_scores    List[float]  – per-frame optical-flow magnitudes
#   brightness_scores List[float] – per-frame mean brightness


class Analyzer:
    """Analyses a video file and returns a structured analysis result dict.

    Parameters
    ----------
    config:
        System-wide :class:`~utils.config.Config` instance.
    model_manager:
        Optional pre-instantiated :class:`~ai_engine.models.ModelManager`.
        A new one is created if not supplied.
    """

    def __init__(
        self,
        config: Config,
        model_manager: Optional[ModelManager] = None,
    ) -> None:
        self.cfg = config
        self.models = model_manager or ModelManager(use_gpu=config.use_gpu)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, video_path: str) -> Dict:
        """Run the full analysis pipeline on *video_path*.

        Returns
        -------
        dict
            An *AnalysisResult* dictionary (see module docstring for keys).
        """
        logger.info("Starting analysis of '%s'", video_path)
        video_info = get_video_info(video_path)
        logger.info(
            "Video: %dx%d  %.2f fps  %s duration",
            video_info["width"],
            video_info["height"],
            video_info["fps"],
            format_time(video_info["duration"]),
        )

        # ---- Frame-level analysis ----
        (
            scenes,
            motion_scores,
            brightness_scores,
        ) = self._analyze_frames(video_path, video_info)

        logger.info("Detected %d scene(s).", len(scenes))

        # ---- Audio analysis ----
        audio_segments = self._analyze_audio(video_path)
        logger.info(
            "Audio: %d segment(s), %d silence(s).",
            len(audio_segments),
            sum(1 for s in audio_segments if s["is_silence"]),
        )

        # ---- Face detection (sampled per scene) ----
        faces_by_scene: Dict[int, List] = {}
        if self.cfg.enable_face_detection:
            faces_by_scene = self._detect_faces_per_scene(video_path, scenes)

        # ---- Object detection (optional) ----
        objects_by_scene: Dict[int, List] = {}
        if self.cfg.enable_object_detection:
            objects_by_scene = self._detect_objects_per_scene(video_path, scenes)

        result = {
            "video_info": video_info,
            "scenes": scenes,
            "audio_segments": audio_segments,
            "faces_by_scene": faces_by_scene,
            "objects_by_scene": objects_by_scene,
            "motion_scores": motion_scores,
            "brightness_scores": brightness_scores,
        }
        logger.info("Analysis complete.")
        return result

    # ------------------------------------------------------------------
    # Frame analysis
    # ------------------------------------------------------------------

    def _analyze_frames(
        self,
        video_path: str,
        video_info: Dict,
    ) -> Tuple[List[Dict], List[float], List[float]]:
        """Detect scenes and collect per-frame motion & brightness scores."""
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError("OpenCV (cv2) is required for frame analysis.") from exc

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = video_info["fps"] or 25.0
        total_frames = video_info["frame_count"]
        threshold = self.cfg.scene_threshold  # 0-1 normalised diff score

        prev_gray: Optional[np.ndarray] = None
        motion_scores: List[float] = []
        brightness_scores: List[float] = []

        # Scene boundary detection
        scene_boundaries: List[int] = [0]  # frame indices where new scenes start
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness_scores.append(float(np.mean(gray)))

            if prev_gray is not None:
                # Frame differencing score (0-1)
                diff = cv2.absdiff(gray, prev_gray)
                diff_score = float(np.mean(diff)) / 255.0
                motion_scores.append(diff_score)

                if diff_score > threshold:
                    scene_boundaries.append(frame_idx)
            else:
                motion_scores.append(0.0)

            prev_gray = gray
            frame_idx += 1

        cap.release()

        # Convert boundaries to scene dicts
        scene_boundaries.append(frame_idx)  # sentinel end
        scenes = self._boundaries_to_scenes(
            scene_boundaries, fps, motion_scores, brightness_scores
        )

        return scenes, motion_scores, brightness_scores

    def _boundaries_to_scenes(
        self,
        boundaries: List[int],
        fps: float,
        motion_scores: List[float],
        brightness_scores: List[float],
    ) -> List[Dict]:
        """Convert a sorted list of boundary frame indices into scene dicts."""
        scenes: List[Dict] = []
        min_frames = int(self.cfg.min_scene_duration * fps)

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]

            if (end - start) < min_frames:
                # Merge short scene into the previous one if possible
                if scenes:
                    scenes[-1]["end_frame"] = end
                    scenes[-1]["end_time"] = end / fps
                    scenes[-1]["duration"] = scenes[-1]["end_time"] - scenes[-1]["start_time"]
                continue

            seg_motion = motion_scores[start:end]
            seg_brightness = brightness_scores[start:end]

            scenes.append(
                {
                    "start_frame": start,
                    "end_frame": end,
                    "start_time": start / fps,
                    "end_time": end / fps,
                    "duration": (end - start) / fps,
                    "avg_motion": float(np.mean(seg_motion)) if seg_motion else 0.0,
                    "avg_brightness": float(np.mean(seg_brightness)) if seg_brightness else 0.0,
                }
            )

        return scenes

    # ------------------------------------------------------------------
    # Audio analysis
    # ------------------------------------------------------------------

    def _analyze_audio(self, video_path: str) -> List[Dict]:
        """Detect silence and active audio segments."""
        try:
            import librosa  # type: ignore
            import soundfile as sf  # type: ignore
        except ImportError:
            logger.warning("librosa/soundfile not available; skipping audio analysis.")
            return []

        # Extract audio to a temporary WAV file using a secure named temp file
        tmp_fd, tmp_audio = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_fd)  # close the fd; FFmpeg will write to the path
        try:
            self._extract_audio(video_path, tmp_audio)
            if not os.path.exists(tmp_audio) or os.path.getsize(tmp_audio) == 0:
                return []

            y, sr = librosa.load(tmp_audio, sr=None, mono=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Audio loading failed: %s", exc)
            return []
        finally:
            if os.path.exists(tmp_audio):
                os.remove(tmp_audio)

        frame_length = int(sr * 0.05)  # 50 ms frames
        hop_length = frame_length // 2

        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        # Convert to dBFS (add small epsilon to avoid log(0))
        rms_db = 20.0 * np.log10(rms + 1e-9)
        frame_times = librosa.frames_to_time(
            np.arange(len(rms_db)), sr=sr, hop_length=hop_length
        )

        segments: List[Dict] = []
        threshold = float(self.cfg.silence_threshold_db)
        min_silence = float(self.cfg.min_silence_duration)

        in_silence = rms_db[0] < threshold
        seg_start = 0.0

        for i, (t, db) in enumerate(zip(frame_times, rms_db)):
            is_silence_now = db < threshold
            if is_silence_now != in_silence:
                duration = t - seg_start
                # Only keep silence segments long enough to remove
                if in_silence and duration < min_silence:
                    pass  # too short – treat as active
                else:
                    segments.append(
                        {
                            "start_time": seg_start,
                            "end_time": t,
                            "is_silence": in_silence,
                            "rms_db": float(np.mean(rms_db[max(0, i - 10) : i + 1])),
                        }
                    )
                seg_start = t
                in_silence = is_silence_now

        # Final segment
        total_dur = float(frame_times[-1]) if len(frame_times) > 0 else 0.0
        if seg_start < total_dur:
            segments.append(
                {
                    "start_time": seg_start,
                    "end_time": total_dur,
                    "is_silence": in_silence,
                    "rms_db": float(np.mean(rms_db[-10:])),
                }
            )

        return segments

    def _extract_audio(self, video_path: str, audio_path: str) -> None:
        """Use FFmpeg to extract audio from *video_path* to *audio_path* (WAV)."""
        import subprocess

        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vn",
            "-ac", "1",
            "-ar", "22050",
            "-f", "wav",
            audio_path,
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            logger.warning("FFmpeg audio extraction returned non-zero exit code.")

    # ------------------------------------------------------------------
    # Face detection
    # ------------------------------------------------------------------

    def _detect_faces_per_scene(
        self, video_path: str, scenes: List[Dict]
    ) -> Dict[int, List]:
        """Sample one frame per scene and detect faces in it."""
        try:
            import cv2  # type: ignore
        except ImportError:
            return {}

        detector = self.models.get_face_detector(self.cfg.face_detection_model)
        cap = cv2.VideoCapture(video_path)
        faces_by_scene: Dict[int, List] = {}

        for idx, scene in enumerate(scenes):
            # Sample the middle frame of each scene
            mid_frame = (scene["start_frame"] + scene["end_frame"]) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
            ret, frame = cap.read()
            if not ret:
                faces_by_scene[idx] = []
                continue
            faces_by_scene[idx] = detector.detect(frame)

        cap.release()
        return faces_by_scene

    # ------------------------------------------------------------------
    # Object detection
    # ------------------------------------------------------------------

    def _detect_objects_per_scene(
        self, video_path: str, scenes: List[Dict]
    ) -> Dict[int, List]:
        """Sample one frame per scene and run object detection."""
        try:
            import cv2  # type: ignore
        except ImportError:
            return {}

        detector = self.models.get_object_detector()
        cap = cv2.VideoCapture(video_path)
        objects_by_scene: Dict[int, List] = {}

        for idx, scene in enumerate(scenes):
            mid_frame = (scene["start_frame"] + scene["end_frame"]) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
            ret, frame = cap.read()
            if not ret:
                objects_by_scene[idx] = []
                continue
            objects_by_scene[idx] = detector.detect(frame)

        cap.release()
        return objects_by_scene
