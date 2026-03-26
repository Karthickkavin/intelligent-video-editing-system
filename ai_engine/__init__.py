"""
AI Engine package for the Intelligent Video Editing System.

Exposes the three core pipeline stages:

    Analyzer  →  VideoEditor  →  Renderer

Quick usage::

    from ai_engine import Analyzer, VideoEditor, Renderer
    from utils.config import Config

    cfg = Config(quality="720p", intensity="medium")
    analyzer  = Analyzer(cfg)
    editor    = VideoEditor(cfg)
    renderer  = Renderer(cfg)

    analysis  = analyzer.analyze("input.mp4")
    edit_plan = editor.create_edit_plan(analysis)
    renderer.render("input.mp4", edit_plan, "output.mp4")
"""

from ai_engine.analyzer import Analyzer
from ai_engine.editor import VideoEditor
from ai_engine.renderer import Renderer
from ai_engine.models import ModelManager

__all__ = ["Analyzer", "VideoEditor", "Renderer", "ModelManager"]
