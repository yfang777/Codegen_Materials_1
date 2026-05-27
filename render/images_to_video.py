from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


def natural_key(path: Path) -> list[int | str]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part for part in parts]


def collect_frames(pattern: str) -> list[Path]:
    frames = [Path(path) for path in glob.glob(pattern)]
    frames.sort(key=natural_key)
    if not frames:
        raise FileNotFoundError(f"No images matched: {pattern}")
    return frames


def images_to_video(
    frame_pattern: str,
    output_path: Path,
    fps: int,
    backend: str,
    quality: int,
    crf: int,
    preset: str,
    macro_block_size: int,
) -> None:
    frames = collect_frames(frame_pattern)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if backend in {"auto", "ffmpeg"} and shutil.which("ffmpeg") is not None:
        images_to_video_ffmpeg(frames, output_path, fps, crf, preset)
    elif backend == "ffmpeg":
        raise FileNotFoundError("ffmpeg was requested, but it was not found on PATH.")
    else:
        images_to_video_imageio(frames, output_path, fps, quality, macro_block_size)


def images_to_video_ffmpeg(frames: list[Path], output_path: Path, fps: int, crf: int, preset: str) -> None:
    with tempfile.TemporaryDirectory(prefix="images_to_video_") as tmp:
        tmp_dir = Path(tmp)
        extension = frames[0].suffix.lower() or ".png"
        for idx, frame_path in enumerate(frames, start=1):
            link_path = tmp_dir / f"frame_{idx:06d}{extension}"
            try:
                os.symlink(frame_path.resolve(), link_path)
            except OSError:
                shutil.copy2(frame_path, link_path)

        input_pattern = tmp_dir / f"frame_%06d{extension}"
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(input_pattern),
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        subprocess.run(cmd, check=True)

    print(f"Saved {output_path} from {len(frames)} frames at {fps} fps using ffmpeg.")


def images_to_video_imageio(
    frames: list[Path],
    output_path: Path,
    fps: int,
    quality: int,
    macro_block_size: int,
) -> None:
    import imageio.v2 as imageio

    first_frame = imageio.imread(frames[0])
    expected_shape = first_frame.shape
    with imageio.get_writer(
        output_path,
        fps=fps,
        quality=quality,
        macro_block_size=macro_block_size,
    ) as writer:
        writer.append_data(first_frame)
        for frame_path in frames[1:]:
            frame = imageio.imread(frame_path)
            if frame.shape != expected_shape:
                raise ValueError(
                    f"{frame_path} has shape {frame.shape}, expected {expected_shape}. "
                    "All frames must have the same resolution."
            )
            writer.append_data(frame)

    print(f"Saved {output_path} from {len(frames)} frames at {fps} fps using imageio.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose rendered image frames into a video.")
    parser.add_argument(
        "--frames",
        required=True,
        help="Glob for rendered frames, for example 'path/to/sample.png*.png'. Quote this so the shell does not expand it.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--backend", choices=["auto", "ffmpeg", "imageio"], default="auto")
    parser.add_argument(
        "--quality",
        type=int,
        default=8,
        help="Imageio quality from 0 to 10. Higher is better and larger.",
    )
    parser.add_argument("--crf", type=int, default=18, help="FFmpeg H.264 CRF. Lower is better and larger.")
    parser.add_argument(
        "--preset",
        default="medium",
        help="FFmpeg x264 preset, for example ultrafast, veryfast, medium, or slow.",
    )
    parser.add_argument(
        "--macro-block-size",
        type=int,
        default=1,
        help="Use 1 to preserve arbitrary frame sizes without resizing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_to_video(
        frame_pattern=args.frames,
        output_path=args.output,
        fps=args.fps,
        backend=args.backend,
        quality=args.quality,
        crf=args.crf,
        preset=args.preset,
        macro_block_size=args.macro_block_size,
    )


if __name__ == "__main__":
    main()
