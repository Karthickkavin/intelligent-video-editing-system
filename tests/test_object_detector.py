"""Tests for the ObjectDetector module."""

import unittest

import numpy as np

from src.config import Config
from src.object_detector import Detection, ObjectDetector, Track


def _blank_frame(h=64, w=64, color=(128, 128, 128)):
    frame = np.full((h, w, 3), color, dtype=np.uint8)
    return frame


class TestDetection(unittest.TestCase):
    def test_center(self):
        det = Detection(
            frame_number=0,
            label="face",
            confidence=0.9,
            bbox=(10, 20, 40, 60),
        )
        cx, cy = det.center
        self.assertAlmostEqual(cx, 30.0)
        self.assertAlmostEqual(cy, 50.0)

    def test_str_representation(self):
        det = Detection(frame_number=5, label="face", confidence=0.8, bbox=(0, 0, 10, 10))
        s = str(det)
        self.assertIn("face", s)
        self.assertIn("5", s)


class TestTrack(unittest.TestCase):
    def test_start_end_frame(self):
        track = Track(track_id=0, label="face")
        d1 = Detection(frame_number=5, label="face", confidence=1.0, bbox=(0, 0, 10, 10))
        d2 = Detection(frame_number=15, label="face", confidence=1.0, bbox=(5, 5, 10, 10))
        track.detections = [d1, d2]
        self.assertEqual(track.start_frame, 5)
        self.assertEqual(track.end_frame, 15)

    def test_trajectory_length(self):
        track = Track(track_id=1, label="face")
        for i in range(5):
            track.detections.append(
                Detection(frame_number=i, label="face", confidence=1.0, bbox=(i, i, 10, 10))
            )
        self.assertEqual(len(track.trajectory), 5)

    def test_empty_track(self):
        track = Track(track_id=2, label="face")
        self.assertEqual(track.start_frame, 0)
        self.assertEqual(track.end_frame, 0)


class TestObjectDetector(unittest.TestCase):
    def setUp(self):
        self.config = Config(
            face_detection_enabled=True,
            object_detection_enabled=False,
            object_confidence_threshold=0.5,
        )
        self.detector = ObjectDetector(self.config)

    def tearDown(self):
        self.detector.close()

    def test_detect_frame_returns_list(self):
        frame = _blank_frame()
        detections = self.detector.detect_frame(frame, frame_number=0)
        self.assertIsInstance(detections, list)

    def test_detect_sequence_returns_dict(self):
        frames = [(i, _blank_frame()) for i in range(5)]
        results = self.detector.detect_sequence(frames)
        self.assertIsInstance(results, dict)

    def test_iou_identical_boxes(self):
        box = (10, 10, 50, 50)
        iou = ObjectDetector._iou(box, box)
        self.assertAlmostEqual(iou, 1.0)

    def test_iou_non_overlapping(self):
        box1 = (0, 0, 10, 10)
        box2 = (20, 20, 10, 10)
        iou = ObjectDetector._iou(box1, box2)
        self.assertAlmostEqual(iou, 0.0)

    def test_iou_partial_overlap(self):
        box1 = (0, 0, 20, 20)
        box2 = (10, 10, 20, 20)
        iou = ObjectDetector._iou(box1, box2)
        self.assertGreater(iou, 0.0)
        self.assertLess(iou, 1.0)

    def test_track_objects_returns_list(self):
        frame_detections = {
            0: [Detection(frame_number=0, label="face", confidence=0.9, bbox=(5, 5, 20, 20))],
            1: [Detection(frame_number=1, label="face", confidence=0.9, bbox=(6, 6, 20, 20))],
        }
        tracks = self.detector.track_objects(frame_detections)
        self.assertIsInstance(tracks, list)
        self.assertEqual(len(tracks), 1)  # same object across 2 frames → 1 track

    def test_track_objects_creates_new_track_for_distant_box(self):
        frame_detections = {
            0: [Detection(frame_number=0, label="face", confidence=0.9, bbox=(0, 0, 10, 10))],
            1: [Detection(frame_number=1, label="face", confidence=0.9, bbox=(100, 100, 10, 10))],
        }
        tracks = self.detector.track_objects(frame_detections)
        # No IoU overlap → two separate tracks
        self.assertEqual(len(tracks), 2)

    def test_generate_suggestions_returns_list(self):
        track = Track(track_id=0, label="face")
        for i in range(10):
            track.detections.append(
                Detection(frame_number=i, label="face", confidence=0.9, bbox=(0, 0, 10, 10))
            )
        suggestions = self.detector.generate_suggestions([track], fps=25.0)
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)

    def test_annotate_frame_returns_array(self):
        frame = _blank_frame()
        detections = [
            Detection(frame_number=0, label="face", confidence=0.9, bbox=(5, 5, 20, 20))
        ]
        annotated = self.detector.annotate_frame(frame, detections)
        self.assertIsInstance(annotated, np.ndarray)
        self.assertEqual(annotated.shape, frame.shape)

    def test_context_manager(self):
        with ObjectDetector(self.config) as detector:
            result = detector.detect_frame(_blank_frame(), 0)
        self.assertIsInstance(result, list)

    def test_disabled_detection_returns_empty(self):
        config = Config(face_detection_enabled=False, object_detection_enabled=False)
        detector = ObjectDetector(config)
        detections = detector.detect_frame(_blank_frame(), 0)
        self.assertEqual(detections, [])
        detector.close()


if __name__ == "__main__":
    unittest.main()
