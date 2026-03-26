import os
from flask import Flask
from flask_cors import CORS


def create_app():
    app = Flask(__name__)

    CORS(app, origins='*')

    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['OUTPUT_FOLDER'] = 'outputs'
    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

    from app.routes.video_routes import video_bp
    app.register_blueprint(video_bp, url_prefix='/')

    return app
