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


def set_material_render_alpha_options(material: bpy.types.Material) -> None:
    material.diffuse_color[3] = min(max(material.diffuse_color[3], 0.0), 1.0)
    if hasattr(material, "blend_method"):
        material.blend_method = "BLEND"
    if hasattr(material, "show_transparent_back"):
        material.show_transparent_back = True
    if hasattr(material, "surface_render_method"):
        try:
            material.surface_render_method = "BLENDED"
        except TypeError:
            pass


def transparent_material(
    source: bpy.types.Material | None,
    alpha: float,
    cache: dict[tuple[str, float], bpy.types.Material],
) -> bpy.types.Material:
    source_name = source.name if source is not None else "default"
    key = (source_name, alpha)
    if key in cache:
        return cache[key]

    if source is None:
        material = bpy.data.materials.new(f"multipose_{source_name}_{alpha:.2f}")
        material.diffuse_color = (0.8, 0.8, 0.8, alpha)
        material.use_nodes = True
        principled = material.node_tree.nodes.get("Principled BSDF")
        if principled is not None:
            if "Alpha" in principled.inputs:
                principled.inputs["Alpha"].default_value = alpha
            if "Base Color" in principled.inputs:
                principled.inputs["Base Color"].default_value = material.diffuse_color
        set_material_render_alpha_options(material)
        cache[key] = material
        return material

    material = source.copy()
    material.name = f"{source.name}_multipose_alpha_{alpha:.2f}"
    material.diffuse_color = (
        material.diffuse_color[0],
        material.diffuse_color[1],
        material.diffuse_color[2],
        alpha,
    )
    set_material_render_alpha_options(material)

    if not material.use_nodes or material.node_tree is None:
        cache[key] = material
        return material

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    output = next((node for node in nodes if node.type == "OUTPUT_MATERIAL"), None)
    if output is None or "Surface" not in output.inputs:
        cache[key] = material
        return material

    surface = output.inputs["Surface"]
    original_socket = surface.links[0].from_socket if surface.is_linked else None
    if original_socket is None:
        cache[key] = material
        return material

    for link in list(surface.links):
        links.remove(link)

    transparent = nodes.new(type="ShaderNodeBsdfTransparent")
    mixer = nodes.new(type="ShaderNodeMixShader")
    mixer.inputs[0].default_value = 1.0 - alpha
    links.new(original_socket, mixer.inputs[1])
    links.new(transparent.outputs["BSDF"], mixer.inputs[2])
    links.new(mixer.outputs["Shader"], surface)

    cache[key] = material
    return material


def matrix_is_close(a: bpy.types.Object, b: bpy.types.Object, tolerance: float) -> bool:
    for row in range(4):
        for col in range(4):
            if abs(a[row][col] - b[row][col]) > tolerance:
                return False
    return True


def evaluated_matrix(obj: bpy.types.Object) -> bpy.types.Object:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    return obj.evaluated_get(depsgraph).matrix_world.copy()


def object_moves(obj: bpy.types.Object, blender_frames: list[int], tolerance: float) -> bool:
    first_matrix = None
    scene = bpy.context.scene
    for frame in blender_frames:
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        matrix = evaluated_matrix(obj)
        if first_matrix is None:
            first_matrix = matrix
        elif not matrix_is_close(first_matrix, matrix, tolerance):
            return True
    return False


def duplicate_evaluated_object(
    obj: bpy.types.Object,
    name: str,
    alpha: float,
    material_cache: dict[tuple[str, float], bpy.types.Material],
) -> bpy.types.Object:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = bpy.data.meshes.new_from_object(evaluated, depsgraph=depsgraph)
    mesh.name = f"{name}_Mesh"

    duplicate = bpy.data.objects.new(name, mesh)
    duplicate.matrix_world = evaluated.matrix_world.copy()
    bpy.context.collection.objects.link(duplicate)

    if alpha < 0.999:
        if mesh.materials:
            for idx, material in enumerate(mesh.materials):
                mesh.materials[idx] = transparent_material(material, alpha, material_cache)
        else:
            mesh.materials.append(transparent_material(None, alpha, material_cache))

    duplicate.hide_viewport = False
    duplicate.hide_render = False
    return duplicate


def source_mesh_objects() -> list[bpy.types.Object]:
    return [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH" and obj.name != "Plane" and not obj.hide_render
    ]


def render_multipose(args: argparse.Namespace) -> None:
    bpy.ops.wm.open_mainfile(filepath=str(args.blend))
    scene = bpy.context.scene

    if args.engine != "BASE":
        scene.render.engine = args.engine
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = True
        configure_cycles_device(args.cycles_render_device)

    scene.render.film_transparent = args.transparent_background
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA" if args.transparent_background else "RGB"
    scene.render.resolution_percentage = 100
    scene.render.use_file_extension = True

    if args.hide_plane:
        plane = bpy.data.objects.get("Plane")
        if plane is not None:
            plane.hide_viewport = True
            plane.hide_render = True

    blender_frames = [step + 1 if args.zero_based else step for step in args.steps]
    sources = source_mesh_objects()
    material_cache: dict[tuple[str, float], bpy.types.Material] = {}

    moving_sources = []
    static_sources = []
    for source in sources:
        if object_moves(source, blender_frames, args.static_tolerance):
            moving_sources.append(source)
        else:
            static_sources.append(source)

    first_frame = blender_frames[0]
    scene.frame_set(first_frame)
    bpy.context.view_layer.update()
    for source in static_sources:
        duplicate_evaluated_object(
            source,
            f"multipose_static_{source.name}",
            args.static_alpha,
            material_cache,
        )

    for step, blender_frame in zip(args.steps, blender_frames, strict=False):
        scene.frame_set(blender_frame)
        bpy.context.view_layer.update()
        for source in moving_sources:
            duplicate_evaluated_object(
                source,
                f"multipose_step_{step:03d}_{source.name}",
                args.alpha,
                material_cache,
            )

    for source in sources:
        source.hide_viewport = True
        source.hide_render = True

    scene.frame_set(first_frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(args.output)
    bpy.ops.render.render(write_still=True)
    print(
        f"Rendered {args.output} from {len(args.steps)} poses "
        f"({len(moving_sources)} moving meshes, {len(static_sources)} static meshes)."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render multiple animation poses together in one Blender scene."
    )
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--steps", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--zero-based",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat --steps as rollout indices. Blender frame = step + 1.",
    )
    parser.add_argument("--alpha", type=float, default=0.38)
    parser.add_argument("--static-alpha", type=float, default=1.0)
    parser.add_argument("--static-tolerance", type=float, default=1e-5)
    parser.add_argument("--transparent-background", action="store_true")
    parser.add_argument("--hide-plane", action="store_true")
    parser.add_argument("--engine", choices=["BASE", "CYCLES", "BLENDER_EEVEE"], default="BASE")
    parser.add_argument("--cycles-render-device", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--samples", type=int, default=64)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be between 0 and 1.")
    if not 0.0 <= args.static_alpha <= 1.0:
        parser.error("--static-alpha must be between 0 and 1.")
    return args


if __name__ == "__main__":
    render_multipose(parse_args())
