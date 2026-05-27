from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROLLOUT = ROOT / "data" / "success_rollout" / "hang_mug" / "sample_0.parquet"
DEFAULT_CAMERA = "camera_239222303153"
BASE_BLEND_URL = "https://www.cs.columbia.edu/~huy/assets/umi_on_legs.blend"
DEFAULT_BASE_BLEND = ROOT / "assets" / "blender" / "umi_on_legs.blend"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def check_export_python() -> None:
    check = "import mujoco, numpy, pandas, pyarrow"
    result = subprocess.run([sys.executable, "-c", check], capture_output=True, text=True)
    if result.returncode == 0:
        return
    raise SystemExit(
        "The export stage needs mujoco, numpy, pandas, and pyarrow in the Python "
        f"environment running this wrapper. Current interpreter: {sys.executable}\n"
        "Run this with your codegen/rollout environment, or use --skip-export with "
        "an existing --animation-pkl."
    )


def ensure_base_blend(base_blend: Path) -> None:
    if base_blend.exists() and base_blend.stat().st_size > 0:
        return
    if base_blend != DEFAULT_BASE_BLEND:
        raise SystemExit(f"Base blend file does not exist: {base_blend}")

    base_blend.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Stanford starter blend to {base_blend}", flush=True)
    print(f"Source: {BASE_BLEND_URL}", flush=True)
    urllib.request.urlretrieve(BASE_BLEND_URL, base_blend)


def blender_version(blender: str) -> tuple[int, int, int] | None:
    result = subprocess.run([blender, "--background", "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        return None
    match = re.search(r"Blender\s+(\d+)\.(\d+)\.(\d+)", result.stdout + result.stderr)
    if not match:
        return None
    return tuple(int(group) for group in match.groups())


def check_base_blend_compat(blender: str, base_blend: Path) -> None:
    if base_blend != DEFAULT_BASE_BLEND:
        return
    version = blender_version(blender)
    if version is None:
        return
    if version < (3, 6, 0):
        raise SystemExit(
            f"{base_blend} was saved with Blender 3.6, but {blender} is Blender "
            f"{version[0]}.{version[1]}.{version[2]}. Use --blender with Blender 3.6+ "
            "for the Stanford base scene, or pass --no-base-blend."
        )


def resolve_blender(blender: str, base_blend: Path | None) -> str:
    if base_blend != DEFAULT_BASE_BLEND:
        return blender

    version = blender_version(blender)
    if version is None or version >= (3, 6, 0):
        return blender

    for candidate in ("/snap/bin/blender", "blender-5.0", "blender-4.0", "blender-3.6"):
        candidate_path = shutil.which(candidate) if not candidate.startswith("/") else candidate
        if candidate_path is None or not Path(candidate_path).exists():
            continue
        candidate_version = blender_version(candidate_path)
        if candidate_version is not None and candidate_version >= (3, 6, 0):
            print(
                f"Using {candidate_path} for the Stanford base blend "
                f"(plain {blender} is Blender {version[0]}.{version[1]}.{version[2]}).",
                flush=True,
            )
            return candidate_path

    check_base_blend_compat(blender, base_blend)
    return blender


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a MuJoCo parquet rollout and build a Blender .blend scene."
    )
    parser.add_argument("--rollout", type=Path, default=DEFAULT_ROLLOUT)
    parser.add_argument("--scene-dir", type=Path, default=None)
    parser.add_argument("--objects", nargs="*", default=None)
    parser.add_argument("--animation-pkl", type=Path, default=None)
    parser.add_argument("--base-blend", type=Path, default=DEFAULT_BASE_BLEND)
    parser.add_argument(
        "--no-base-blend",
        action="store_true",
        help="Build a plain scene instead of using the Stanford starter .blend.",
    )
    parser.add_argument("--blend-out", type=Path, default=None)
    parser.add_argument("--render-output", type=Path, default=None)
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--engine", choices=["BASE", "EEVEE", "CYCLES"], default="BASE")
    parser.add_argument("--cycles-render-device", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--camera-mode", choices=["base", "mujoco"], default="base")
    parser.add_argument("--camera-focus", choices=["objects", "scene"], default="objects")
    parser.add_argument("--camera-margin", type=float, default=None)
    parser.add_argument("--object-camera-margin", type=float, default=1.24)
    parser.add_argument("--scene-camera-margin", type=float, default=1.72)
    parser.add_argument("--camera-yaw-offset-deg", type=float, default=-70.0)
    parser.add_argument("--freestyle-mode", choices=["thin", "off", "base"], default="off")
    parser.add_argument("--freestyle-thickness", type=float, default=0.4)
    parser.add_argument("--transparent-background", action="store_true")
    parser.add_argument("--shadow-catcher", action="store_true")
    parser.add_argument("--no-keep-base-stage", action="store_true")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument(
        "--geom-groups",
        nargs="*",
        type=int,
        default=None,
        help="Optional MuJoCo geom groups to export. Defaults to every visible geom.",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Reuse --animation-pkl instead of regenerating it.",
    )
    args = parser.parse_args()

    if args.animation_pkl is None:
        args.animation_pkl = args.rollout.with_name(f"{args.rollout.stem}_blender_animation.pkl")
    if args.blend_out is None:
        args.blend_out = args.rollout.with_name(f"{args.rollout.stem}_blender.blend")
    if args.no_base_blend:
        args.base_blend = None
    if args.skip_export and not args.animation_pkl.exists():
        parser.error(f"--skip-export was set, but {args.animation_pkl} does not exist.")
    return args


def main() -> None:
    args = parse_args()

    if not args.skip_export:
        check_export_python()
        export_cmd = [
            sys.executable,
            str(ROOT / "render" / "export_blender_animation.py"),
            "--rollout",
            str(args.rollout),
            "--output",
            str(args.animation_pkl),
            "--fps",
            str(args.fps),
            "--stride",
            str(args.stride),
            "--camera",
            args.camera,
        ]
        if args.width is not None:
            export_cmd.extend(["--width", str(args.width)])
        if args.height is not None:
            export_cmd.extend(["--height", str(args.height)])
        if args.scene_dir is not None:
            export_cmd.extend(["--scene-dir", str(args.scene_dir)])
        if args.objects:
            export_cmd.append("--objects")
            export_cmd.extend(args.objects)
        if args.max_frames is not None:
            export_cmd.extend(["--max-frames", str(args.max_frames)])
        if args.geom_groups is not None:
            export_cmd.append("--geom-groups")
            export_cmd.extend(str(group) for group in args.geom_groups)
        run(export_cmd)

    if args.base_blend is not None:
        ensure_base_blend(args.base_blend)
        args.blender = resolve_blender(args.blender, args.base_blend)

    blender_cmd = [
        args.blender,
        "--background",
        "--python",
        str(ROOT / "render" / "create_blender_scene.py"),
        "--",
        "--animation-pkl",
        str(args.animation_pkl),
        "--blend-out",
        str(args.blend_out),
        "--engine",
        args.engine,
        "--cycles-render-device",
        args.cycles_render_device,
        "--samples",
        str(args.samples),
        "--fps",
        str(args.fps),
        "--clear-base-objects",
        "--camera-mode",
        args.camera_mode,
        "--camera-focus",
        args.camera_focus,
        "--object-camera-margin",
        str(args.object_camera_margin),
        "--scene-camera-margin",
        str(args.scene_camera_margin),
        "--camera-yaw-offset-deg",
        str(args.camera_yaw_offset_deg),
        "--freestyle-mode",
        args.freestyle_mode,
        "--freestyle-thickness",
        str(args.freestyle_thickness),
    ]
    if args.transparent_background:
        blender_cmd.append("--transparent-background")
    if args.shadow_catcher:
        blender_cmd.append("--shadow-catcher")
    if args.camera_margin is not None:
        blender_cmd.extend(["--camera-margin", str(args.camera_margin)])
    if args.width is not None:
        blender_cmd.extend(["--width", str(args.width)])
    if args.height is not None:
        blender_cmd.extend(["--height", str(args.height)])
    if args.no_keep_base_stage:
        blender_cmd.append("--no-keep-base-stage")
    if args.base_blend is not None:
        blender_cmd.extend(["--base-blend", str(args.base_blend)])
    if args.render_output is not None:
        blender_cmd.extend(["--render-output", str(args.render_output)])
    run(blender_cmd)


if __name__ == "__main__":
    main()
