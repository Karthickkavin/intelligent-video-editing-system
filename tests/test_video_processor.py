"""Tests for the VideoProcessor module."""

import os
import tempfile
import unittest

import cv2
import numpy as np

from src.config import Config
from src.video_processor import VideoClip, VideoMetadata, VideoProcessor


def _create_test_video(path: str, num_frames: int = 30, fps: float = 25.0) -> None:
    """Write a minimal synthetic video to *path*."""
    width, height = 64, 64
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    rng = np.random.default_rng(42)
    for _ in range(num_frames):
        frame = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


class TestVideoMetadata(unittest.TestCase):
    def test_str_representation(self):
        meta = VideoMetadata(
            path="/tmp/test.mp4",
            width=1920,
            height=1080,
            fps=30.0,
            total_frames=900,
            duration=30.0,
        )
        s = str(meta)
        self.assertIn("test.mp4", s)
        self.assertIn("1920x1080", s)


class TestVideoClip(unittest.TestCase):
    def test_timestamps(self):
        clip = VideoClip(start_frame=0, end_frame=50, fps=25.0)
        self.assertAlmostEqual(clip.start_time, 0.0)
        self.assertAlmostEqual(clip.end_time, 2.0)
        self.assertAlmostEqual(clip.duration, 2.0)

    def test_str_representation(self):
        clip = VideoClip(start_frame=25, end_frame=75, fps=25.0, label="test")
        s = str(clip)
        self.assertIn("test", s)


class TestVideoProcessor(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.video_path = os.path.join(self.tmp_dir, "test.mp4")
        _create_test_video(self.video_path, num_frames=30, fps=25.0)

    def test_load_returns_metadata(self):
        config = Config()
        proc = VideoProcessor(config)
        meta = proc.load(self.video_path)
        self.assertIsInstance(meta, VideoMetadata)
        self.assertEqual(meta.path, self.video_path)
        self.assertEqual(meta.width, 64)
        self.assertEqual(meta.height, 64)
        self.assertAlmostEqual(meta.fps, 25.0, places=0)
        proc.release()

    def test_load_nonexistent_raises(self):
        proc = VideoProcessor()
        with self.assertRaises(FileNotFoundError):
            proc.load("/nonexistent/path/video.mp4")

    def test_frames_generator(self):
        with VideoProcessor() as proc:
            proc.load(self.video_path)
            frames = list(proc.frames(skip=1))
        self.assertEqual(len(frames), 30)
        for idx, frame in frames:
            self.assertIsInstance(idx, int)
            self.assertIsInstance(frame, np.ndarray)

    def test_frames_skip(self):
        with VideoProcessor() as proc:
            proc.load(self.video_path)
            frames = list(proc.frames(skip=5))
        # With 30 frames and skip=5, we expect frame indices 0,5,10,15,20,25
        self.assertEqual(len(frames), 6)

    def test_read_frame(self):
        with VideoProcessor() as proc:
            proc.load(self.video_path)
            frame = proc.read_frame(0)
        self.assertIsNotNone(frame)
        self.assertIsInstance(frame, np.ndarray)

    def test_extract_frames(self):
        out_dir = os.path.join(self.tmp_dir, "frames")
        with VideoProcessor() as proc:
            proc.load(self.video_path)
            paths = proc.extract_frames(out_dir, skip=5)
        self.assertTrue(len(paths) > 0)
        for p in paths:
            self.assertTrue(os.path.isfile(p))

    def test_save_clip(self):
        out_path = os.path.join(self.tmp_dir, "clip.mp4")
        with VideoProcessor() as proc:
            meta = proc.load(self.video_path)
            clip = VideoClip(
                start_frame=0, end_frame=10, fps=meta.fps, label="clip"
            )
            success = proc.save_clip(clip, out_path)
        self.assertTrue(success)
        self.assertTrue(os.path.isfile(out_path))

    def test_context_manager(self):
        with VideoProcessor() as proc:
            meta = proc.load(self.video_path)
            self.assertIsNotNone(meta)
        # After exiting the context, the capture should be released
        self.assertIsNone(proc._cap)

    def test_metadata_property(self):
        proc = VideoProcessor()
        self.assertIsNone(proc.metadata)
        proc.load(self.video_path)
        self.assertIsNotNone(proc.metadata)
        proc.release()

    def test_max_frames_config(self):
        config = Config(max_frames=10)
        with VideoProcessor(config) as proc:
            proc.load(self.video_path)
            frames = list(proc.frames())
        self.assertLessEqual(len(frames), 10)


if __name__ == "__main__":
    unittest.main()
