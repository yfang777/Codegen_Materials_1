from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy


def configure_cycles_device(requested: str) -> None:
    scene = bpy.context.scene
    if scene.render.engine != "CYCLES":
        return

    requested = requested.lower()
    if requested == "cpu":
        scene.cycles.device = "CPU"
        print("Using Cycles CPU rendering.")
        return

    prefs = bpy.context.preferences.addons.get("cycles")
    cycles_prefs = prefs.preferences if prefs is not None else None
    device_types = ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI")

    if cycles_prefs is not None:
        for device_type in device_types:
            try:
                cycles_prefs.compute_device_type = device_type
                cycles_prefs.get_devices()
            except (AttributeError, TypeError, RuntimeError):
                continue

            devices = list(cycles_prefs.devices)
            gpu_devices = [device for device in devices if getattr(device, "type", "") != "CPU"]
            if not gpu_devices:
                continue

            for device in devices:
                device.use = device in gpu_devices
            scene.cycles.device = "GPU"
            names = ", ".join(device.name for device in gpu_devices)
            print(f"Using Cycles GPU rendering via {device_type}: {names}")
            return

    if requested == "gpu":
        raise RuntimeError("Cycles GPU rendering was requested, but Blender found no usable GPU device.")

    scene.cycles.device = "CPU"
    print("No Cycles GPU device found; falling back to CPU rendering.")


def configure_transparency(shadow_catcher: bool) -> None:
    scene = bpy.context.scene
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.use_file_extension = True

    plane = bpy.data.objects.get("Plane")
    if shadow_catcher:
        if scene.render.engine != "CYCLES":
            raise RuntimeError("--shadow-catcher requires Cycles rendering.")
        if plane is None:
            print("Warning: --shadow-catcher was requested, but no Plane object exists.")
            return
        plane.hide_viewport = False
        plane.hide_render = False
        if hasattr(plane, "is_shadow_catcher"):
            plane.is_shadow_catcher = True
        elif hasattr(plane, "cycles") and hasattr(plane.cycles, "is_shadow_catcher"):
            plane.cycles.is_shadow_catcher = True
        else:
            print("Warning: this Blender build does not expose a shadow catcher property on Plane.")
    elif plane is not None:
        plane.hide_render = True


def output_path_for_step(output_dir: Path, prefix: str, step: int) -> Path:
    return output_dir / f"{prefix}{step:03d}.png"


def render_steps(args: argparse.Namespace) -> None:
    bpy.ops.wm.open_mainfile(filepath=str(args.blend))
    scene = bpy.context.scene

    if args.engine != "BASE":
        scene.render.engine = args.engine
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = True
        configure_cycles_device(args.cycles_render_device)

    configure_transparency(args.shadow_catcher)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    original_frame = scene.frame_current
    for step in args.steps:
        blender_frame = step + 1 if args.zero_based else step
        out_path = output_path_for_step(args.output_dir, args.prefix, step)
        scene.frame_set(blender_frame)
        scene.render.filepath = str(out_path)
        bpy.ops.render.render(write_still=True)
        print(f"Rendered step {step} as Blender frame {blender_frame}: {out_path}")

    scene.frame_set(original_frame)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render selected transparent PNG frames from an existing Blender animation."
    )
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--steps", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="alpha_fixed_step_")
    parser.add_argument(
        "--zero-based",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat --steps as rollout indices. Blender frame = step + 1.",
    )
    parser.add_argument("--engine", choices=["BASE", "CYCLES", "BLENDER_EEVEE"], default="BASE")
    parser.add_argument("--cycles-render-device", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--shadow-catcher", action="store_true")
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


if __name__ == "__main__":
    render_steps(parse_args())
