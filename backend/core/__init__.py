"""Core package for the Autonomous Smart Video Editor."""

from .video_processor import VideoProcessor, VideoSegment
from .decision_engine import DecisionEngine
from .renderer import Renderer

__all__ = ["VideoProcessor", "VideoSegment", "DecisionEngine", "Renderer"]
