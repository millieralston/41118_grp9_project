#!/usr/bin/env python3
"""
Watch the training preview image (preview_frame.png) and save periodic screenshots,
then assemble them into a video using ffmpeg when the watcher is stopped.

Usage:
  python scripts/record_preview.py --out recordings/training_preview.mp4 --fps 5

Run this in a separate terminal while `python train.py` is running.
"""
import argparse
import os
import shutil
import time
import subprocess
from pathlib import Path

PREVIEW_PATH = Path("preview_frame.png")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def watch_and_record(frames_dir: Path, interval: float):
    ensure_dir(frames_dir)
    counter = 0
    last_mtime = None
    print(f"Watching {PREVIEW_PATH} (interval={interval}s). Press Ctrl-C to stop and assemble video.")
    try:
        while True:
            if PREVIEW_PATH.exists():
                mtime = PREVIEW_PATH.stat().st_mtime
                if last_mtime is None or mtime != last_mtime:
                    last_mtime = mtime
                    dest = frames_dir / f"frame_{counter:06d}.png"
                    try:
                        shutil.copy2(PREVIEW_PATH, dest)
                        print(f"Saved {dest}")
                        counter += 1
                    except Exception as e:
                        print(f"Failed to copy preview frame: {e}")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return counter


def assemble_video(frames_dir: Path, out_path: Path, fps: int):
    if not frames_dir.exists():
        print(f"No frames directory {frames_dir}")
        return False
    pattern = str(frames_dir / "frame_%06d.png")
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    print("Running:", " ".join(cmd))
    try:
        subprocess.check_call(cmd)
        print(f"Saved video to {out_path}")
        return True
    except FileNotFoundError:
        print("ffmpeg not found. Install ffmpeg or assemble frames manually:")
        print("ffmpeg -y -framerate <fps> -i frame_%06d.png -c:v libx264 -pix_fmt yuv420p out.mp4")
        return False
    except subprocess.CalledProcessError as e:
        print("ffmpeg failed:", e)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", default="recordings/frames", help="Directory to save frames")
    parser.add_argument("--out", default="recordings/training_preview.mp4", help="Output MP4 path")
    parser.add_argument("--interval", type=float, default=0.5, help="Polling interval in seconds")
    parser.add_argument("--fps", type=int, default=5, help="FPS for assembled video")

    args = parser.parse_args()
    frames_dir = Path(args.frames_dir)
    out_path = Path(args.out)
    ensure_dir(frames_dir.parent)

    saved = watch_and_record(frames_dir, args.interval)
    if saved > 0:
        assemble_video(frames_dir, out_path, args.fps)
    else:
        print("No frames were saved; no video assembled.")


if __name__ == "__main__":
    main()
