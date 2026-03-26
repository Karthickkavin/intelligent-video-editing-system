"""
Example usage of the Intelligent Video Editing System.

This script demonstrates the full workflow:
1. Load a video file
2. Detect scenes and key frames
3. Analyse motion patterns
4. Detect objects/faces
5. Generate an editing plan
6. Export selected clips

Usage:
    python examples/example_usage.py --video path/to/video.mp4 --output output/
"""

import argparse
import logging
import os
import sys

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.auto_editor import AutoEditor
from src.config import Config, EditingProfile
from src.motion_analyzer import MotionAnalyzer
from src.object_detector import ObjectDetector
from src.scene_detector import SceneDetector
from src.utils import format_timestamp, setup_logging
from src.video_processor import VideoProcessor

logger = logging.getLogger("video_editing_system.example")


def run_example(video_path: str, output_dir: str, profile_name: str = "highlights") -> None:
    """Run the complete video editing pipeline on *video_path*."""

    # ── 1. Configuration ──────────────────────────────────────────────────────
    config = Config(output_dir=output_dir)
    profile_map = {
        "fast_paced": EditingProfile.FAST_PACED,
        "documentary": EditingProfile.DOCUMENTARY,
        "highlights": EditingProfile.HIGHLIGHTS,
        "summary": EditingProfile.SUMMARY,
    }
    profile = profile_map.get(profile_name.lower(), EditingProfile.HIGHLIGHTS)
    config.apply_profile(profile)
    logger.info("Using profile: %s", profile.value)

    # ── 2. Load video ─────────────────────────────────────────────────────────
    with VideoProcessor(config) as processor:
        logger.info("Loading video: %s", video_path)
        metadata = processor.load(video_path)
        print(f"\n{'='*60}")
        print(f"  Video: {os.path.basename(video_path)}")
        print(f"  Resolution: {metadata.width}x{metadata.height}")
        print(f"  FPS: {metadata.fps:.2f}")
        print(f"  Duration: {format_timestamp(metadata.duration)}")
        print(f"  Frames: {metadata.total_frames}")
        print(f"  Size: {metadata.file_size_mb:.1f} MB")
        print(f"{'='*60}\n")

        # Collect frames (sub-sample for speed in this demo)
        sample_skip = max(1, metadata.total_frames // 300)
        logger.info("Collecting frames (skip=%d) …", sample_skip)
        frames = list(processor.frames(skip=sample_skip))
        logger.info("Collected %d sample frames", len(frames))

        # ── 3. Scene detection ────────────────────────────────────────────────
        print("[ SCENE DETECTION ]")
        scene_detector = SceneDetector(config)
        scenes = scene_detector.detect(frames, fps=metadata.fps)
        print(f"  Detected {len(scenes)} scenes")
        for i, scene in enumerate(scenes[:5], 1):
            print(
                f"  Scene {i:>3}: {format_timestamp(scene.start_time)} → "
                f"{format_timestamp(scene.end_time)} "
                f"(key frame: {scene.key_frame})"
            )
        if len(scenes) > 5:
            print(f"  … and {len(scenes) - 5} more scenes")
        print()

        # ── 4. Motion analysis ────────────────────────────────────────────────
        print("[ MOTION ANALYSIS ]")
        motion_analyzer = MotionAnalyzer(config)
        motion_results = motion_analyzer.analyze(frames, fps=metadata.fps)
        motion_segments = motion_analyzer.segment(motion_results, fps=metadata.fps)
        highlights = [s for s in motion_segments if s.is_highlight]
        print(f"  Motion segments: {len(motion_segments)}")
        print(f"  Highlight segments: {len(highlights)}")
        for seg in highlights[:3]:
            print(
                f"  Highlight: {format_timestamp(seg.start_time)} → "
                f"{format_timestamp(seg.end_time)} "
                f"(intensity={seg.mean_intensity:.2f})"
            )
        print()

        # ── 5. Object / face detection ────────────────────────────────────────
        print("[ OBJECT & FACE DETECTION ]")
        with ObjectDetector(config) as obj_detector:
            # Limit to at most 50 frames for the demo
            detection_frames = frames[:: max(1, len(frames) // 50)]
            frame_detections = obj_detector.detect_sequence(detection_frames)
            tracks = obj_detector.track_objects(frame_detections)
            suggestions = obj_detector.generate_suggestions(tracks, fps=metadata.fps)
            print(f"  Frames with detections: {len(frame_detections)}")
            print(f"  Object tracks: {len(tracks)}")
            for s in suggestions:
                print(f"  ▸ {s}")
        print()

        # ── 6. Auto-editing plan ──────────────────────────────────────────────
        print("[ AUTO-EDITING PLAN ]")
        with AutoEditor(config) as auto_editor:
            plan = auto_editor.generate_plan(frames, fps=metadata.fps)
        print(f"  Cut suggestions: {len(plan.suggestions)}")
        print(f"  Highlights: {len(plan.highlights)}")
        print(f"  Summary segments: {len(plan.summary_segments)}")
        print(
            f"  Summary duration: {format_timestamp(plan.summary_duration)} "
            f"(of {format_timestamp(plan.total_duration)})"
        )
        print()

        # ── 7. Export clips ───────────────────────────────────────────────────
        if plan.summary_segments:
            print("[ EXPORTING SUMMARY CLIPS ]")
            clips_dir = os.path.join(output_dir, "summary_clips")
            os.makedirs(clips_dir, exist_ok=True)

            from src.video_processor import VideoClip

            clips = [
                VideoClip(
                    start_frame=seg.start_frame,
                    end_frame=seg.end_frame,
                    fps=metadata.fps,
                    label=f"summary_{i:03d}",
                    score=seg.score,
                )
                for i, seg in enumerate(plan.summary_segments)
            ]
            saved = processor.save_clips(clips, clips_dir)
            print(f"  Exported {len(saved)} clips to: {clips_dir}")
            for p in saved:
                print(f"  ✓ {os.path.basename(p)}")
        print()
        print("Done! ✔")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Intelligent Video Editing System — example usage"
    )
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument(
        "--output", default="output", help="Output directory (default: output/)"
    )
    parser.add_argument(
        "--profile",
        default="highlights",
        choices=["fast_paced", "documentary", "highlights", "summary"],
        help="Editing profile to use",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    args = parser.parse_args()

    setup_logging(level=args.log_level)

    if not os.path.isfile(args.video):
        print(f"Error: video file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    run_example(args.video, args.output, args.profile)


if __name__ == "__main__":
    main()
