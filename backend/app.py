"""
Flask application factory for the Autonomous Smart Video Editor.
"""

import os
import sys

from flask import Flask, send_from_directory

from config import get_config
from api import api_bp
from utils.helpers import ensure_directory
from utils.logger import get_logger

logger = get_logger(__name__)


def create_app(env: str = None) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__, static_folder="static")

    cfg = get_config(env)
    app.config.from_object(cfg)

    # Store the processor config object so routes can access it
    app.config["PROCESSOR_CONFIG"] = cfg

    ensure_directory(app.config["UPLOAD_FOLDER"])
    ensure_directory(app.config["OUTPUT_FOLDER"])

    app.register_blueprint(api_bp)

    # Serve the web UI for all non-API routes
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve_ui(path):  # noqa: ANN001
        static_dir = os.path.join(app.root_path, "static")
        if path and os.path.exists(os.path.join(static_dir, path)):
            return send_from_directory(static_dir, path)
        return send_from_directory(static_dir, "index.html")

    logger.info("Application created (env=%s)", env or "default")
    return app
