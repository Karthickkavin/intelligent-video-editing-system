"""
Example usage of the Intelligent Video Editing System.

Run this script to process a local video file::

    python example_usage.py --input path/to/video.mp4 --quality 720p --intensity medium

Or import the pipeline into your own code:

    from example_usage import run_pipeline
    run_pipeline("input.mp4", "output.mp4")
"""

import argparse
import logging
import os
import sys
from typing import Optional

# Make sure the project root is on the path when running directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config import Config
from utils.logger import setup_logger
from utils.helpers import validate_video_file, format_time
from ai_engine.analyzer import Analyzer
from ai_engine.editor import VideoEditor
from ai_engine.renderer import Renderer


# ---------------------------------------------------------------------------
# High-level pipeline function
# ---------------------------------------------------------------------------


def run_pipeline(
    input_path: str,
    output_path: Optional[str] = None,
    quality: str = "720p",
    intensity: str = "medium",
    enable_object_detection: bool = False,
    progress_callback=None,
) -> str:
    """Run the full AI video editing pipeline.

    Parameters
    ----------
    input_path:
        Path to the source video (MP4, MOV, AVI, etc.).
    output_path:
        Destination path.  Defaults to ``output/<input_name>_edited.mp4``.
    quality:
        One of ``"480p"``, ``"720p"``, ``"1080p"``.
    intensity:
        One of ``"light"``, ``"medium"``, ``"heavy"``.
    enable_object_detection:
        Set to ``True`` to run YOLOv5 object detection (slower, GPU recommended).
    progress_callback:
        Optional ``(pct: float, msg: str) -> None`` callable.

    Returns
    -------
    str
        Absolute path of the rendered output video.
    """
    # ---- Validate input ----
    ok, msg = validate_video_file(input_path)
    if not ok:
        raise ValueError(f"Invalid input: {msg}")

    # ---- Default output path ----
    if output_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join("output", f"{base}_edited.mp4")

    # ---- Configuration ----
    cfg = Config(
        quality=quality,
        intensity=intensity,
        enable_object_detection=enable_object_detection,
    )

    logger = logging.getLogger("video_editor")
    logger.info("=" * 60)
    logger.info("Intelligent Video Editing System")
    logger.info("Input  : %s", input_path)
    logger.info("Output : %s", output_path)
    logger.info("Quality: %s  Intensity: %s", quality, intensity)
    logger.info("=" * 60)

    # ---- Stage 1: Analyze ----
    if progress_callback:
        progress_callback(0, "Starting analysis …")

    analyzer = Analyzer(cfg)
    analysis = analyzer.analyze(input_path)

    if progress_callback:
        progress_callback(30, "Analysis complete. Planning edits …")

    # ---- Stage 2: Edit ----
    editor = VideoEditor(cfg)
    edit_plan = editor.create_edit_plan(analysis)

    if progress_callback:
        progress_callback(40, "Edit plan ready. Rendering …")

    # ---- Stage 3: Render ----
    def _render_progress(pct, msg):
        # Map renderer's 0-100 into 40-100 of overall progress
        overall = 40 + int(pct * 0.6)
        if progress_callback:
            progress_callback(overall, msg)

    renderer = Renderer(cfg)
    output = renderer.render(
        input_path,
        edit_plan,
        output_path,
        progress_callback=_render_progress,
    )

    logger.info("Done! Output saved to: %s", output)
    return output


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def run_batch(
    input_paths: list,
    output_dir: str = "output",
    **pipeline_kwargs,
) -> list:
    """Process multiple videos sequentially.

    Parameters
    ----------
    input_paths:
        List of video file paths to process.
    output_dir:
        Directory where all edited videos are saved.
    **pipeline_kwargs:
        Extra keyword arguments forwarded to :func:`run_pipeline`.

    Returns
    -------
    list of str
        Paths of all successfully rendered output files.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for i, path in enumerate(input_paths, 1):
        print(f"\n[{i}/{len(input_paths)}] Processing: {path}")
        base = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(output_dir, f"{base}_edited.mp4")
        try:
            result = run_pipeline(path, out, **pipeline_kwargs)
            results.append(result)
            print(f"  ✓ Saved: {result}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ Failed: {exc}")

    return results


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Intelligent Video Editing System – CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", "-o", default=None, help="Output video path")
    parser.add_argument(
        "--quality",
        "-q",
        choices=["480p", "720p", "1080p"],
        default="720p",
        help="Output quality preset",
    )
    parser.add_argument(
        "--intensity",
        choices=["light", "medium", "heavy"],
        default="medium",
        help="Editing intensity level",
    )
    parser.add_argument(
        "--object-detection",
        action="store_true",
        help="Enable YOLOv5 object detection (requires torch + internet)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    setup_logger(log_level=getattr(logging, args.log_level))

    try:
        output = run_pipeline(
            input_path=args.input,
            output_path=args.output,
            quality=args.quality,
            intensity=args.intensity,
            enable_object_detection=args.object_detection,
            progress_callback=lambda pct, msg: print(f"[{pct:3.0f}%] {msg}"),
        )
        print(f"\n✅ Success! Output: {output}")
        sys.exit(0)
    except Exception as exc:
        print(f"\n❌ Error: {exc}", file=sys.stderr)
        sys.exit(1)
