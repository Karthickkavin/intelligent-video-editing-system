"""
Configuration module for the Intelligent Video Editing System.
Provides configurable sensitivity levels, editing profiles, and output options.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class OutputFormat(Enum):
    MP4 = "mp4"
    AVI = "avi"
    MOV = "mov"
    WEBM = "webm"


class EditingProfile(Enum):
    FAST_PACED = "fast_paced"
    DOCUMENTARY = "documentary"
    HIGHLIGHTS = "highlights"
    SUMMARY = "summary"
    CUSTOM = "custom"


@dataclass
class Config:
    """Central configuration for the video editing system."""

    # Scene detection settings
    scene_threshold: float = 0.4
    min_scene_length: int = 15  # minimum frames per scene
    histogram_bins: int = 256

    # Object detection settings
    object_confidence_threshold: float = 0.5
    face_detection_enabled: bool = True
    object_detection_enabled: bool = True

    # Motion analysis settings
    motion_threshold: float = 0.3
    optical_flow_levels: int = 3
    motion_blur_kernel: int = 5

    # Auto-editing settings
    editing_profile: EditingProfile = EditingProfile.HIGHLIGHTS
    highlight_motion_threshold: float = 0.6
    summary_ratio: float = 0.2  # fraction of original duration to keep

    # Video processing settings
    target_fps: Optional[int] = None  # None means keep original
    max_frames: Optional[int] = None  # None means process all frames
    frame_skip: int = 1  # process every Nth frame

    # Output settings
    output_format: OutputFormat = OutputFormat.MP4
    output_quality: int = 23  # CRF value (lower = better quality)
    output_dir: str = "output"

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # Editing profiles presets
    PROFILES: dict = field(default_factory=lambda: {
        EditingProfile.FAST_PACED: {
            "scene_threshold": 0.3,
            "min_scene_length": 10,
            "highlight_motion_threshold": 0.5,
            "summary_ratio": 0.3,
        },
        EditingProfile.DOCUMENTARY: {
            "scene_threshold": 0.5,
            "min_scene_length": 30,
            "highlight_motion_threshold": 0.7,
            "summary_ratio": 0.5,
        },
        EditingProfile.HIGHLIGHTS: {
            "scene_threshold": 0.4,
            "min_scene_length": 15,
            "highlight_motion_threshold": 0.6,
            "summary_ratio": 0.2,
        },
        EditingProfile.SUMMARY: {
            "scene_threshold": 0.35,
            "min_scene_length": 20,
            "highlight_motion_threshold": 0.55,
            "summary_ratio": 0.1,
        },
    })

    def apply_profile(self, profile: EditingProfile) -> None:
        """Apply a preset editing profile to the current configuration."""
        self.editing_profile = profile
        preset = self.PROFILES.get(profile, {})
        for key, value in preset.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Create a Config instance from a dictionary."""
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
