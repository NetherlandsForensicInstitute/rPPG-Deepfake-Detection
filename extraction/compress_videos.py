import argparse
import subprocess
import sys
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".avi"}

FFMPEG_ARGS = [
    "-vcodec", "libx264",
    "-crf", "28",
    "-pix_fmt", "yuv420p",
    "-r", "30",
    "-an",
]


def compress(src: Path, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["ffmpeg", "-i", str(src), *FFMPEG_ARGS, "-y", str(dst)],
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.decode()[-300:]}", file=sys.stderr)
        dst.unlink(missing_ok=True)
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Compress videos using ffmpeg/libx264")
    parser.add_argument("source", type=Path, help="Source directory containing videos")
    parser.add_argument("output", type=Path, help="Output directory for compressed videos")
    args = parser.parse_args()

    videos = sorted(
        p for p in args.source.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS
    )
    total = len(videos)
    print(f"Found {total} video files\nOutput: {args.output}\n")

    compressed = skipped = failed = 0

    for i, src in enumerate(videos, 1):
        rel = src.relative_to(args.source)
        dst = args.output / rel.with_suffix(".mp4")

        if dst.exists():
            print(f"[{i}/{total}] Skip (exists): {rel}")
            skipped += 1
            continue

        print(f"[{i}/{total}] Compressing: {rel}")
        if compress(src, dst):
            print(f"  -> {dst}")
            compressed += 1
        else:
            print(f"  -> FAILED: {src}", file=sys.stderr)
            failed += 1

    print(f"\nDone: {compressed} compressed, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
