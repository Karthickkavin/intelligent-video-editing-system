import cv2
import numpy as np


class SceneDetector:
    def detect_scenes(self, video_path, threshold=30.0):
        """
        Detect scene changes in a video using frame difference analysis.
        Returns list of (start_time, end_time) tuples in seconds.
        """
        scenes = []
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return []

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 24.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps

            scene_starts = [0.0]
            prev_gray = None
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (320, 240))

                if prev_gray is not None:
                    diff = cv2.absdiff(gray, prev_gray)
                    mean_diff = np.mean(diff)
                    if mean_diff > threshold:
                        timestamp = frame_idx / fps
                        scene_starts.append(timestamp)

                prev_gray = gray
                frame_idx += 1

            cap.release()

            scene_starts.append(duration)

            for i in range(len(scene_starts) - 1):
                start = scene_starts[i]
                end = scene_starts[i + 1]
                if (end - start) >= 0.5:
                    scenes.append((start, end))

        except Exception as e:
            print(f"Scene detection error: {e}")
            return []

        return scenes

    def get_scene_count(self, video_path, threshold=30.0):
        """Returns integer count of detected scenes."""
        try:
            scenes = self.detect_scenes(video_path, threshold)
            return len(scenes)
        except Exception:
            return 0
