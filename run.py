"""
Triple Jump Analysis — CLI entry point.

All pipeline logic lives in src/pipeline.py.
This file is a thin CLI wrapper that passes a print-based progress callback.

Usage:
    python run.py <video_path> [options]

Options:
    --stride N          Pose estimation every N frames (default: 3)
    --start S           Start at second S (default: 0)
    --end E             End at second E (default: full video)
    --max-frames N      Max analysis frames (default: unlimited)
    --no-seg            Skip segmentation/tracking (faster, less accurate)
    --output DIR        Output directory (default: ./output/<video_name>)
    --annotate-every N  Save annotated image every N analysis frames (default: 3)
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from src.pipeline  import PipelineConfig, run_pipeline
from src.job_store import print_progress


def parse_args():
    p = argparse.ArgumentParser(description="Triple Jump video analysis")
    p.add_argument("video")
    p.add_argument("--stride",         type=int,   default=3)
    p.add_argument("--start",          type=float, default=0.0)
    p.add_argument("--end",            type=float, default=None)
    p.add_argument("--max-frames",     type=int,   default=None)
    p.add_argument("--no-seg",         action="store_true")
    p.add_argument("--output",         type=str,   default=None)
    p.add_argument("--annotate-every", type=int,   default=1)
    return p.parse_args()


def main():
    args   = parse_args()
    vname  = Path(args.video).stem
    outdir = args.output or str(Path("output") / vname)

    config = PipelineConfig(
        video_path=args.video,
        output_dir=outdir,
        stride=args.stride,
        start_sec=args.start,
        end_sec=args.end,
        max_frames=args.max_frames,
        use_seg=not args.no_seg,
        annotate_every=args.annotate_every,
    )

    print(f"\n[Input]  {args.video}")
    print(f"[Output] {outdir}\n")

    import warnings
    warnings.filterwarnings("ignore")

    summary = run_pipeline(config, on_progress=print_progress)

    print(f"\n{'═'*55}")
    print("  SUMMARY")
    print(f"{'═'*55}")
    if "error" in summary:
        print(f"  ERROR: {summary['error']}")
        return

    print(f"  Frames analyzed     : {summary['total_frames_analyzed']}")
    print(f"  Person detected     : {summary['frames_with_person']} ({summary['detection_rate_pct']}%)")
    print(f"  Usable for analysis : {summary['frames_usable_for_analysis']} ({summary['usable_rate_pct']}%)")
    print(f"  Dominant angle      : {summary['dominant_angle']}")
    print(f"  Quality avg         : {summary['quality_score']['mean']:.3f}")
    print()
    for angle, pct in sorted(summary["camera_angle_distribution"].items(), key=lambda x: -x[1]):
        bar = "█" * int(pct / 3)
        tag = "  ← best" if angle == "LATERAL" else ""
        print(f"    {angle:<12} {pct:5.1f}%  {bar}{tag}")
    print(f"{'═'*55}\n")


if __name__ == "__main__":
    main()
