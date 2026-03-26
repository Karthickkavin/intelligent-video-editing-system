"""
Entry point for the Autonomous Smart Video Editor backend.

Usage:
    python run.py [--env development|production] [--host 0.0.0.0] [--port 5000]
"""

import argparse
import sys
import os

# Ensure the backend directory is on the path when run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from utils.helpers import ffmpeg_available, ffprobe_available
from utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous Smart Video Editor")
    parser.add_argument("--env", default="development", choices=["development", "production"])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    if not ffmpeg_available():
        logger.error("FFmpeg is not available on PATH. Please install FFmpeg.")
        sys.exit(1)

    if not ffprobe_available():
        logger.error("FFprobe is not available on PATH. Please install FFmpeg.")
        sys.exit(1)

    app = create_app(env=args.env)
    logger.info("Starting server on %s:%d (env=%s)", args.host, args.port, args.env)
    app.run(host=args.host, port=args.port, debug=(args.env == "development"))


if __name__ == "__main__":
    main()
