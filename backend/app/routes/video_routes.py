import os
import uuid
import threading
from datetime import datetime
from flask import Blueprint, request, jsonify, send_file, current_app
from werkzeug.utils import secure_filename
from app.services.video_processor import process_video

video_bp = Blueprint('video', __name__)

ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'webm'}
jobs = {}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@video_bp.route('/api/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format. Supported: MP4, MOV, AVI, MKV, WebM'}), 400

    job_id = str(uuid.uuid4())
    ext = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{job_id}.{ext}"

    upload_folder = current_app.config['UPLOAD_FOLDER']
    output_folder = current_app.config['OUTPUT_FOLDER']

    input_path = os.path.join(upload_folder, filename)
    output_path = os.path.join(output_folder, f"edited_{job_id}.mp4")

    file.save(input_path)

    jobs[job_id] = {
        'id': job_id,
        'status': 'queued',
        'progress': 0,
        'message': 'Job queued...',
        'input_file': input_path,
        'result_filename': None,
        'created_at': datetime.utcnow().isoformat(),
        'settings': {}
    }

    thread = threading.Thread(
        target=process_video,
        args=(job_id, input_path, output_path, jobs)
    )
    thread.daemon = True
    thread.start()

    return jsonify({'job_id': job_id, 'status': 'queued'}), 202


@video_bp.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    job = jobs[job_id]
    return jsonify({
        'id': job['id'],
        'status': job['status'],
        'progress': job['progress'],
        'message': job['message'],
        'result_filename': job['result_filename'],
        'created_at': job['created_at']
    })


@video_bp.route('/api/download/<job_id>', methods=['GET'])
def download_video(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    job = jobs[job_id]
    if job['status'] != 'completed':
        return jsonify({'error': 'Job not completed yet'}), 400
    output_folder = current_app.config['OUTPUT_FOLDER']
    file_path = os.path.join(output_folder, job['result_filename'])
    if not os.path.exists(file_path):
        return jsonify({'error': 'Output file not found'}), 404
    return send_file(file_path, as_attachment=True, download_name=job['result_filename'])


@video_bp.route('/api/jobs', methods=['GET'])
def list_jobs():
    job_list = []
    for job in jobs.values():
        job_list.append({
            'id': job['id'],
            'status': job['status'],
            'progress': job['progress'],
            'message': job['message'],
            'created_at': job['created_at'],
            'result_filename': job['result_filename']
        })
    job_list.sort(key=lambda x: x['created_at'], reverse=True)
    return jsonify({'jobs': job_list})
