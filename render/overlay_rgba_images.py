from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def read_rgba(path: Path) -> np.ndarray:
    image = imageio.imread(path)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[2] == 3:
        alpha = np.full(image.shape[:2] + (1,), 255, dtype=image.dtype)
        image = np.concatenate([image, alpha], axis=2)
    if image.shape[2] != 4:
        raise ValueError(f"{path} has unsupported shape {image.shape}; expected RGB or RGBA.")
    return image.astype(np.float32) / 255.0


def alpha_composite(dst: np.ndarray, src: np.ndarray, opacity: float) -> np.ndarray:
    src = src.copy()
    src[:, :, 3:4] *= opacity
    src_alpha = src[:, :, 3:4]
    dst_alpha = dst[:, :, 3:4]
    out_alpha = src_alpha + dst_alpha * (1.0 - src_alpha)
    premultiplied_rgb = src[:, :, :3] * src_alpha + dst[:, :, :3] * dst_alpha * (1.0 - src_alpha)
    out_rgb = np.divide(
        premultiplied_rgb,
        out_alpha,
        out=np.zeros_like(premultiplied_rgb),
        where=out_alpha > 1e-6,
    )
    return np.concatenate([out_rgb, out_alpha], axis=2)


def write_rgba(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8)
    imageio.imwrite(path, image)


def overlay_images(
    image_paths: list[Path],
    output: Path,
    background: Path | None,
    background_color: list[float] | None,
    opacity: float,
) -> None:
    if not image_paths:
        raise ValueError("At least one image is required.")

    if background_color is not None:
        first = read_rgba(image_paths[0])
        canvas = np.ones_like(first)
        alpha = background_color[3] if len(background_color) == 4 else 1.0
        canvas[:, :, 0] = background_color[0]
        canvas[:, :, 1] = background_color[1]
        canvas[:, :, 2] = background_color[2]
        canvas[:, :, 3] = alpha
    elif background is None:
        first = read_rgba(image_paths[0])
        canvas = np.zeros_like(first)
    else:
        canvas = read_rgba(background)

    expected_shape = canvas.shape
    for image_path in image_paths:
        image = read_rgba(image_path)
        if image.shape != expected_shape:
            raise ValueError(f"{image_path} has shape {image.shape}, expected {expected_shape}.")
        canvas = alpha_composite(canvas, image, opacity)

    write_rgba(output, canvas)
    print(f"Saved {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alpha-composite multiple RGB/RGBA images into one PNG.")
    parser.add_argument("--images", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--background", type=Path, default=None)
    parser.add_argument(
        "--background-color",
        nargs="+",
        type=float,
        default=None,
        help="Solid background as 0-1 floats: R G B [A]. Example: --background-color 1 1 1",
    )
    parser.add_argument("--opacity", type=float, default=1.0)
    args = parser.parse_args()
    if args.background is not None and args.background_color is not None:
        parser.error("Use either --background or --background-color, not both.")
    if args.background_color is not None and len(args.background_color) not in {3, 4}:
        parser.error("--background-color expects R G B or R G B A.")
    if args.background_color is not None and any(
        channel < 0.0 or channel > 1.0 for channel in args.background_color
    ):
        parser.error("--background-color values must be between 0 and 1.")
    if not 0.0 <= args.opacity <= 1.0:
        parser.error("--opacity must be between 0 and 1.")
    return args


def main() -> None:
    args = parse_args()
    overlay_images(args.images, args.output, args.background, args.background_color, args.opacity)


if __name__ == "__main__":
    main()
