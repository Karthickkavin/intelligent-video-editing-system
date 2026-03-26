"""Tests for the SceneDetector module."""

import unittest

import numpy as np

from src.config import Config
from src.scene_detector import Scene, SceneDetector


def _make_frames(colors, frame_size=(64, 64)):
    """
    Build a list of (frame_number, ndarray) from a list of BGR colour tuples.
    Each colour tuple produces a solid-colour frame.
    """
    frames = []
    for i, color in enumerate(colors):
        frame = np.full((frame_size[1], frame_size[0], 3), color, dtype=np.uint8)
        frames.append((i, frame))
    return frames


class TestScene(unittest.TestCase):
    def test_properties(self):
        scene = Scene(start_frame=0, end_frame=24, fps=25.0, key_frame=12)
        self.assertAlmostEqual(scene.start_time, 0.0)
        self.assertAlmostEqual(scene.end_time, 0.96)
        self.assertEqual(scene.frame_count, 24)

    def test_str_representation(self):
        scene = Scene(start_frame=0, end_frame=50, fps=25.0)
        s = str(scene)
        self.assertIn("0", s)
        self.assertIn("50", s)


class TestSceneDetector(unittest.TestCase):
    def _make_config(self, threshold=0.3):
        return Config(scene_threshold=threshold, min_scene_length=2)

    def test_empty_frames(self):
        detector = SceneDetector()
        scenes = detector.detect([])
        self.assertEqual(scenes, [])

    def test_single_frame(self):
        detector = SceneDetector(self._make_config())
        frames = _make_frames([(100, 100, 100)])
        scenes = detector.detect(frames)
        self.assertEqual(len(scenes), 1)

    def test_detects_scene_change(self):
        """A sharp change from black to white frames should produce ≥2 scenes."""
        config = self._make_config(threshold=0.1)
        detector = SceneDetector(config)

        black_frames = _make_frames([(0, 0, 0)] * 10)
        white_frames = _make_frames([(255, 255, 255)] * 10)
        # Renumber white_frames
        white_frames = [(10 + i, f) for i, (_, f) in enumerate(white_frames)]
        frames = black_frames + white_frames

        scenes = detector.detect(frames, fps=25.0)
        self.assertGreaterEqual(len(scenes), 2)

    def test_uniform_video_single_scene(self):
        """A video with no scene change should produce one scene."""
        config = self._make_config(threshold=0.5)
        detector = SceneDetector(config)
        frames = _make_frames([(128, 64, 32)] * 20)
        scenes = detector.detect(frames, fps=25.0)
        self.assertEqual(len(scenes), 1)

    def test_scores_populated(self):
        config = self._make_config()
        detector = SceneDetector(config)
        frames = _make_frames([(i * 10, 0, 0) for i in range(10)])
        detector.detect(frames)
        self.assertEqual(len(detector.scores), len(frames))

    def test_detect_transitions(self):
        config = self._make_config(threshold=0.1)
        detector = SceneDetector(config)
        black = _make_frames([(0, 0, 0)] * 5)
        white = [(5 + i, f) for i, (_, f) in enumerate(_make_frames([(255, 255, 255)] * 5))]
        frames = black + white
        transitions = detector.detect_transitions(frames, fps=25.0)
        # At least one transition should be detected at the boundary
        self.assertGreater(len(transitions), 0)

    def test_scene_fps(self):
        config = self._make_config(threshold=0.1)
        detector = SceneDetector(config)
        frames = _make_frames([(0, 0, 0)] * 5 + [(255, 255, 255)] * 5)
        frames[5:] = [(5 + i, f) for i, (_, f) in enumerate(frames[5:])]
        scenes = detector.detect(frames, fps=30.0)
        for scene in scenes:
            self.assertAlmostEqual(scene.fps, 30.0)

    def test_key_frame_in_range(self):
        config = self._make_config(threshold=0.1)
        detector = SceneDetector(config)
        frames = _make_frames([(128, 128, 128)] * 15)
        scenes = detector.detect(frames, fps=25.0)
        for scene in scenes:
            self.assertGreaterEqual(scene.key_frame, scene.start_frame)
            self.assertLessEqual(scene.key_frame, scene.end_frame)


if __name__ == "__main__":
    unittest.main()
