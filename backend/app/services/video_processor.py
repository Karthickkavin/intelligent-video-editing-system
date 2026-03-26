import os
import cv2
import numpy as np
from app.services.scene_detector import SceneDetector
from app.services.audio_sync import AudioSynchronizer


def process_video(job_id, input_path, output_path, jobs):
    """
    Main video processing pipeline.
    Analyzes, detects scenes, trims, adds transitions, syncs audio, and encodes.
    """
    try:
        jobs[job_id]['status'] = 'processing'
        jobs[job_id]['progress'] = 5
        jobs[job_id]['message'] = 'Analyzing video...'

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Cannot open video file")

        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        if duration <= 0:
            raise ValueError("Invalid video duration")

        jobs[job_id]['progress'] = 15
        jobs[job_id]['message'] = 'Detecting scenes...'

        detector = SceneDetector()
        scenes = detector.detect_scenes(input_path, threshold=30.0)

        jobs[job_id]['progress'] = 30
        jobs[job_id]['message'] = 'Removing silent/blank segments...'

        from moviepy.editor import VideoFileClip, concatenate_videoclips

        main_clip = VideoFileClip(input_path)

        jobs[job_id]['progress'] = 50
        jobs[job_id]['message'] = 'Applying smart trimming...'

        valid_clips = []

        if scenes:
            for (start, end) in scenes:
                seg_duration = end - start
                if seg_duration < 0.3:
                    continue
                start = max(0, min(start, main_clip.duration - 0.1))
                end = max(start + 0.1, min(end, main_clip.duration))
                if end - start < 0.3:
                    continue

                try:
                    subclip = main_clip.subclip(start, end)
                    mid_time = (end - start) / 2
                    frame = subclip.get_frame(min(mid_time, subclip.duration - 0.01))
                    mean_brightness = np.mean(frame)
                    if mean_brightness >= 10:
                        valid_clips.append(subclip)
                except Exception:
                    continue

        jobs[job_id]['progress'] = 65
        jobs[job_id]['message'] = 'Adding transitions...'

        if not valid_clips:
            if main_clip.duration > 2.0:
                fallback = main_clip.subclip(0.5, main_clip.duration - 0.5)
                valid_clips = [fallback]
            else:
                valid_clips = [main_clip]

        if len(valid_clips) > 1:
            final_clip = concatenate_videoclips(valid_clips, method="compose")
        else:
            final_clip = valid_clips[0]

        jobs[job_id]['progress'] = 80
        jobs[job_id]['message'] = 'Synchronizing audio...'

        synchronizer = AudioSynchronizer()
        final_clip = synchronizer.apply_audio_effects(final_clip)

        jobs[job_id]['progress'] = 90
        jobs[job_id]['message'] = 'Encoding final video...'

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            fps=24,
            verbose=False,
            logger=None
        )

        main_clip.close()
        final_clip.close()
        for clip in valid_clips:
            try:
                clip.close()
            except Exception:
                pass

        jobs[job_id]['progress'] = 100
        jobs[job_id]['message'] = 'Processing complete!'
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['result_filename'] = os.path.basename(output_path)

    except Exception as e:
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['message'] = str(e)
        jobs[job_id]['progress'] = 0
