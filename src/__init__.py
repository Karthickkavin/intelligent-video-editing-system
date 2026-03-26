"""
Intelligent Video Editing System
AI-powered video editing with scene detection, object tracking, and auto-editing.
"""

from .config import Config, EditingProfile
from .video_processor import VideoProcessor
from .scene_detector import SceneDetector
from .object_detector import ObjectDetector
from .motion_analyzer import MotionAnalyzer
from .auto_editor import AutoEditor

__version__ = "1.0.0"
__all__ = [
    "Config",
    "EditingProfile",
    "VideoProcessor",
    "SceneDetector",
    "ObjectDetector",
    "MotionAnalyzer",
    "AutoEditor",
]
