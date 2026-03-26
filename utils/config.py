"""
Configuration management for the Intelligent Video Editing System.

This module centralises all tunable parameters so that the rest of the code
never hard-codes magic numbers.  Every setting has a sensible default that
works out-of-the-box in Google Colab.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional


# ---------------------------------------------------------------------------
# Quality presets
# ---------------------------------------------------------------------------

QUALITY_PRESETS: Dict[str, Dict] = {
    "480p": {
        "resolution": (854, 480),
        "video_bitrate": "1000k",
        "audio_bitrate": "128k",
        "fps": 24,
        "crf": 28,
    },
    "720p": {
        "resolution": (1280, 720),
        "video_bitrate": "2500k",
        "audio_bitrate": "192k",
        "fps": 30,
        "crf": 23,
    },
    "1080p": {
        "resolution": (1920, 1080),
        "video_bitrate": "5000k",
        "audio_bitrate": "256k",
        "fps": 30,
        "crf": 18,
    },
}


# ---------------------------------------------------------------------------
# Editing intensity presets
# ---------------------------------------------------------------------------

INTENSITY_PRESETS: Dict[str, Dict] = {
    "light": {
        # Scene detection – higher threshold = fewer cuts
        "scene_threshold": 0.6,
        # Minimum scene duration kept (seconds)
        "min_scene_duration": 2.0,
        # Silence threshold (dBFS) – clips quieter than this are removed
        "silence_threshold_db": -45,
        # Minimum silence length to remove (seconds)
        "min_silence_duration": 1.5,
        # Transition length (seconds)
        "transition_duration": 0.5,
        # Speed range applied to clips for emphasis
        "speed_range": (0.9, 1.1),
        # Maximum zoom factor for auto-zoom on faces/objects
        "max_zoom": 1.1,
        # Colour-correction strength 0-1
        "color_correction_strength": 0.3,
        # Audio normalisation target (dBFS)
        "audio_target_db": -18,
    },
    "medium": {
        "scene_threshold": 0.45,
        "min_scene_duration": 1.5,
        "silence_threshold_db": -40,
        "min_silence_duration": 1.0,
        "transition_duration": 0.75,
        "speed_range": (0.75, 1.25),
        "max_zoom": 1.2,
        "color_correction_strength": 0.5,
        "audio_target_db": -16,
    },
    "heavy": {
        "scene_threshold": 0.3,
        "min_scene_duration": 0.8,
        "silence_threshold_db": -35,
        "min_silence_duration": 0.5,
        "transition_duration": 1.0,
        "speed_range": (0.5, 1.5),
        "max_zoom": 1.4,
        "color_correction_strength": 0.8,
        "audio_target_db": -14,
    },
}


# ---------------------------------------------------------------------------
# Main configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """Central configuration object.

    Usage::

        cfg = Config()                        # all defaults
        cfg = Config(intensity="heavy")       # heavy editing
        cfg = Config(quality="1080p",
                     enable_transitions=False) # custom mix
    """

    # ---- Quality ----
    quality: Literal["480p", "720p", "1080p"] = "720p"

    # ---- Intensity ----
    intensity: Literal["light", "medium", "heavy"] = "medium"

    # ---- Feature toggles ----
    enable_scene_detection: bool = True
    enable_silence_removal: bool = True
    enable_transitions: bool = True
    enable_speed_adjustment: bool = True
    enable_auto_zoom: bool = True
    enable_color_correction: bool = True
    enable_audio_normalization: bool = True
    enable_face_detection: bool = True
    enable_object_detection: bool = False  # heavier model – off by default

    # ---- Output ----
    output_format: str = "mp4"
    output_codec: str = "libx264"
    audio_codec: str = "aac"

    # ---- Paths ----
    output_dir: str = "output"
    temp_dir: str = "temp"
    log_dir: str = "logs"

    # ---- Processing limits ----
    max_video_duration: float = 3600.0  # seconds (1 hour)
    max_file_size_mb: float = 2048.0    # 2 GB
    processing_timeout: float = 1800.0  # 30 minutes

    # ---- Model selection ----
    face_detection_model: str = "haarcascade"  # or "mediapipe"
    object_detection_model: str = "yolo"

    # ---- Supported formats ----
    supported_formats: List[str] = field(
        default_factory=lambda: [".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"]
    )

    # ---- Batch processing ----
    batch_max_workers: int = 2

    # ---- Colab / memory ----
    use_gpu: bool = True
    memory_limit_gb: float = 12.0

    # ---- Derived / overridable scene params ----
    # These are populated from the intensity preset in __post_init__
    scene_threshold: Optional[float] = None
    min_scene_duration: Optional[float] = None
    silence_threshold_db: Optional[float] = None
    min_silence_duration: Optional[float] = None
    transition_duration: Optional[float] = None
    speed_range: Optional[tuple] = None
    max_zoom: Optional[float] = None
    color_correction_strength: Optional[float] = None
    audio_target_db: Optional[float] = None

    def __post_init__(self) -> None:
        """Merge intensity preset values for any param not explicitly set."""
        preset = INTENSITY_PRESETS[self.intensity]
        for key, value in preset.items():
            if getattr(self, key) is None:
                setattr(self, key, value)

        # Ensure output directories exist
        for directory in (self.output_dir, self.temp_dir, self.log_dir):
            os.makedirs(directory, exist_ok=True)

    # ---- Convenience accessors ----

    @property
    def quality_settings(self) -> Dict:
        """Return the full quality preset dict."""
        return QUALITY_PRESETS[self.quality]

    @property
    def resolution(self):
        return self.quality_settings["resolution"]

    @property
    def fps(self) -> int:
        return self.quality_settings["fps"]

    @property
    def video_bitrate(self) -> str:
        return self.quality_settings["video_bitrate"]

    @property
    def audio_bitrate(self) -> str:
        return self.quality_settings["audio_bitrate"]

    @property
    def crf(self) -> int:
        return self.quality_settings["crf"]

    def to_dict(self) -> Dict:
        """Return all settings as a plain dictionary (useful for logging)."""
        import dataclasses
        return dataclasses.asdict(self)

    def __repr__(self) -> str:
        return (
            f"Config(quality={self.quality!r}, intensity={self.intensity!r}, "
            f"output_dir={self.output_dir!r})"
        )
