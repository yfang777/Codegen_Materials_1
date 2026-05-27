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


def hide_non_background_objects(background_names: set[str]) -> None:
    for obj in bpy.context.scene.objects:
        if obj.type in {"CAMERA", "LIGHT"} or obj.name in background_names:
            obj.hide_viewport = False
            obj.hide_render = False
            continue
        obj.hide_viewport = True
        obj.hide_render = True


def render_background(args: argparse.Namespace) -> None:
    bpy.ops.wm.open_mainfile(filepath=str(args.blend))
    scene = bpy.context.scene

    if args.engine != "BASE":
        scene.render.engine = args.engine
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = True
        configure_cycles_device(args.cycles_render_device)

    scene.frame_set(args.frame)
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.resolution_percentage = 100
    scene.render.use_file_extension = True
    hide_non_background_objects(set(args.keep_object))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(args.output)
    bpy.ops.render.render(write_still=True)
    print(f"Rendered background {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render only background/stage objects from a Blender file.")
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=1)
    parser.add_argument(
        "--keep-object",
        action="append",
        default=["Plane"],
        help="Object name to keep visible. May be passed multiple times.",
    )
    parser.add_argument("--engine", choices=["BASE", "CYCLES", "BLENDER_EEVEE"], default="BASE")
    parser.add_argument("--cycles-render-device", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--samples", type=int, default=64)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


if __name__ == "__main__":
    render_background(parse_args())
