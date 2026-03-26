"""
Configuration management for the Autonomous Smart Video Editor.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    # Flask settings
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
    MAX_CONTENT_LENGTH = 2 * 1024 * 1024 * 1024  # 2 GB max upload

    # Directory settings
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")

    # Allowed video formats
    ALLOWED_EXTENSIONS = {"mp4", "mov", "mkv", "avi", "webm"}

    # Silence detection defaults
    SILENCE_THRESHOLD = -40          # dB
    SILENCE_MIN_DURATION = 1.5       # seconds

    # Segment scoring thresholds
    MIN_SEGMENT_SCORE = 0.3          # 0.0 – 1.0
    MIN_SEGMENT_DURATION = 2.0       # seconds
    MAX_SEGMENT_DURATION = 60.0      # seconds
    GAP_FILL_THRESHOLD = 0.5         # seconds – merge segments closer than this

    # FFmpeg rendering settings
    VIDEO_PRESET = "fast"
    VIDEO_CRF = 23
    AUDIO_BITRATE = "128k"
    FFMPEG_THREADS = 4

    # Output settings
    OUTPUT_RESOLUTION = None         # None = keep source resolution, e.g. "1280x720"
    OUTPUT_FORMAT = "mp4"

    # Logging
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")


class DevelopmentConfig(Config):
    DEBUG = True
    LOG_LEVEL = "DEBUG"


class ProductionConfig(Config):
    DEBUG = False


config_map = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}


def get_config(env: str = None) -> Config:
    """Return the appropriate config object based on the environment."""
    env = env or os.environ.get("FLASK_ENV", "default")
    return config_map.get(env, DevelopmentConfig)()
