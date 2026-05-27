from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import bpy
from mathutils import Euler, Matrix, Quaternion, Vector


ROOT = Path(__file__).resolve().parents[1]


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def clear_renderable_objects(keep_names: set[str] | None = None) -> None:
    keep_names = keep_names or set()
    for obj in list(bpy.context.scene.objects):
        if obj.name in keep_names:
            continue
        if obj.type in {"ARMATURE", "CURVE", "FONT", "MESH", "META", "SURFACE"}:
            bpy.data.objects.remove(obj, do_unlink=True)


def make_material(name: str, rgba: list[float]) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = rgba
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled:
        principled.inputs["Base Color"].default_value = rgba
        if "Roughness" in principled.inputs:
            principled.inputs["Roughness"].default_value = 0.55
        if "Specular" in principled.inputs:
            principled.inputs["Specular"].default_value = 0.25
        elif "Specular IOR Level" in principled.inputs:
            principled.inputs["Specular IOR Level"].default_value = 0.25
    return material


def set_principled_input(material: bpy.types.Material, input_name: str, value: Any) -> None:
    if not material.use_nodes:
        return
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled and input_name in principled.inputs:
        principled.inputs[input_name].default_value = value


def make_principled_material(
    name: str,
    rgba: list[float],
    roughness: float,
    metallic: float = 0.0,
    specular: float = 0.35,
) -> bpy.types.Material:
    material = make_material(name, rgba)
    set_principled_input(material, "Roughness", roughness)
    set_principled_input(material, "Metallic", metallic)
    set_principled_input(material, "Specular", specular)
    set_principled_input(material, "Specular IOR Level", specular)
    return material


def material_for_visual(
    visual: dict[str, Any],
    material_cache: dict[str, bpy.types.Material],
) -> bpy.types.Material:
    name = str(visual.get("name", "")).lower()
    mesh = str(visual.get("mesh", "")).lower()
    material_key = f"{name}:{mesh}"

    if "finger_pad" in name:
        if "codegen_finger_pad" not in material_cache:
            material_cache["codegen_finger_pad"] = make_principled_material(
                "codegen_finger_pad", [0.035, 0.034, 0.032, 1.0], roughness=0.82, specular=0.16
            )
        return material_cache["codegen_finger_pad"]
    if "mug" in material_key:
        if "codegen_mug_ceramic" not in material_cache:
            material_cache["codegen_mug_ceramic"] = make_principled_material(
                "codegen_mug_ceramic", [0.86, 0.38, 0.24, 1.0], roughness=0.38, specular=0.45
            )
        return material_cache["codegen_mug_ceramic"]
    if "rack" in material_key:
        if "codegen_rack_satin" not in material_cache:
            material_cache["codegen_rack_satin"] = make_principled_material(
                "codegen_rack_satin", [0.72, 0.66, 0.52, 1.0], roughness=0.52, specular=0.28
            )
        return material_cache["codegen_rack_satin"]

    rgba = tuple(round(float(x), 4) for x in visual["rgba"])
    fallback_key = f"rgba:{rgba}"
    if fallback_key not in material_cache:
        material_cache[fallback_key] = make_material(f"mat_{len(material_cache):03d}", list(rgba))
    return material_cache[fallback_key]


def sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def object_xml_mesh_meta(xml_path: Path, mesh_name: str) -> tuple[Path | None, list[float] | None]:
    if not xml_path.exists():
        return None, None

    import xml.etree.ElementTree as ET

    root = ET.parse(xml_path).getroot()
    mesh = root.find(f"./asset/mesh[@name='{mesh_name}']")
    if mesh is None:
        mesh = root.find("./asset/mesh")
    if mesh is None:
        return None, None

    mesh_path = None
    mesh_file = mesh.get("file")
    if mesh_file:
        mesh_path = Path(mesh_file)
        if not mesh_path.is_absolute():
            mesh_path = xml_path.parent / mesh_path
        mesh_path = mesh_path.resolve()

    mesh_scale = None
    if mesh.get("scale"):
        mesh_scale = [float(value) for value in mesh.get("scale", "1 1 1").split()]
        if len(mesh_scale) != 3:
            mesh_scale = None

    return mesh_path, mesh_scale


def find_object_template(object_name: str) -> Path | None:
    matches = sorted((ROOT / "assets").glob(f"objects/**/{object_name}_decomposed_template.xml"))
    return matches[0] if matches else None


def resolve_stl_path(visual: dict[str, Any], payload: dict[str, Any]) -> Path | None:
    if visual.get("stl_path"):
        path = Path(visual["stl_path"]).expanduser()
        return path if path.exists() else None

    mesh_name = visual.get("mesh") or f"{visual['name']}_mesh"
    mesh = payload.get("meshes", {}).get(mesh_name, {})
    if mesh.get("source_path"):
        path = Path(mesh["source_path"]).expanduser()
        if path.exists():
            return path

    source = payload.get("source", {})
    scene_dir = source.get("scene_dir")
    if scene_dir:
        path, _ = object_xml_mesh_meta(Path(scene_dir).expanduser() / f"{visual['name']}.xml", mesh_name)
        if path is not None and path.exists():
            return path

    template = find_object_template(visual["name"])
    if template is not None:
        path, _ = object_xml_mesh_meta(template, mesh_name)
        if path is not None and path.exists():
            return path

    fallback_matches = sorted((ROOT / "assets").glob(f"objects/**/{visual['name']}.stl"))
    return fallback_matches[0] if fallback_matches else None


def resolve_stl_scale(visual: dict[str, Any], payload: dict[str, Any]) -> list[float]:
    if visual.get("stl_scale"):
        return [float(value) for value in visual["stl_scale"]]

    mesh_name = visual.get("mesh") or f"{visual['name']}_mesh"
    source = payload.get("source", {})
    scene_dir = source.get("scene_dir")
    if scene_dir:
        _, scale = object_xml_mesh_meta(Path(scene_dir).expanduser() / f"{visual['name']}.xml", mesh_name)
        if scale is not None:
            return scale

    template = find_object_template(visual["name"])
    if template is not None:
        _, scale = object_xml_mesh_meta(template, mesh_name)
        if scale is not None:
            return scale

    return [1.0, 1.0, 1.0]


def import_stl_object(name: str, path: Path, material: bpy.types.Material) -> bpy.types.Object:
    before = set(bpy.data.objects.keys())
    if hasattr(bpy.ops.wm, "stl_import"):
        bpy.ops.wm.stl_import(filepath=str(path))
    else:
        bpy.ops.import_mesh.stl(filepath=str(path))

    imported = [obj for obj in bpy.context.selected_objects if obj.name not in before]
    if not imported:
        imported = [obj for obj in bpy.data.objects if obj.name not in before]
    if not imported:
        raise RuntimeError(f"Blender did not import any object from {path}")

    obj = imported[0]
    obj.name = name
    obj.data.name = f"{name}_mesh"
    obj.data.materials.clear()
    obj.data.materials.append(material)

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    try:
        bpy.ops.object.shade_smooth()
    except RuntimeError:
        pass
    obj.select_set(False)
    return obj


def make_mesh_object(name: str, vertices: Any, faces: Any, material: bpy.types.Material) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata([tuple(v) for v in vertices], [], [tuple(f) for f in faces])
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    try:
        bpy.ops.object.shade_smooth()
    except RuntimeError:
        pass
    obj.select_set(False)
    return obj


def box_geometry(size: list[float]) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int, int]]]:
    x, y, z = size
    vertices = [
        (-x, -y, -z),
        (x, -y, -z),
        (x, y, -z),
        (-x, y, -z),
        (-x, -y, z),
        (x, -y, z),
        (x, y, z),
        (-x, y, z),
    ]
    faces = [
        (0, 1, 2, 3),
        (4, 7, 6, 5),
        (0, 4, 5, 1),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 4, 0),
    ]
    return vertices, faces


def plane_geometry(size: list[float]) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int, int]]]:
    x = size[0] if len(size) > 0 else 5.0
    y = size[1] if len(size) > 1 else x
    vertices = [(-x, -y, 0.0), (x, -y, 0.0), (x, y, 0.0), (-x, y, 0.0)]
    faces = [(0, 1, 2, 3)]
    return vertices, faces


def sphere_object(name: str, radius: float, material: bpy.types.Material) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=radius)
    obj = bpy.context.object
    obj.name = name
    obj.data.name = f"{name}_mesh"
    obj.data.materials.append(material)
    bpy.ops.object.shade_smooth()
    return obj


def cylinder_object(name: str, radius: float, depth: float, material: bpy.types.Material) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=radius, depth=depth)
    obj = bpy.context.object
    obj.name = name
    obj.data.name = f"{name}_mesh"
    obj.data.materials.append(material)
    bpy.ops.object.shade_smooth()
    return obj


def make_visual_object(
    visual: dict[str, Any],
    payload: dict[str, Any],
    material: bpy.types.Material,
) -> bpy.types.Object | None:
    name = sanitize_name(visual["name"])
    geom_type = visual["type"]

    if geom_type == "mesh":
        if visual.get("source") == "raw_object_mesh":
            stl_path = resolve_stl_path(visual, payload)
            if stl_path is not None:
                visual["stl_scale"] = resolve_stl_scale(visual, payload)
                return import_stl_object(name, stl_path, material)
            print(f"Could not find an STL file for {visual['name']}; falling back to exported mesh vertices.")
        mesh = payload["meshes"][visual["mesh"]]
        return make_mesh_object(name, mesh["vertices"], mesh["faces"], material)

    if geom_type == "box":
        vertices, faces = box_geometry(visual["size"])
        return make_mesh_object(name, vertices, faces, material)

    if geom_type == "plane":
        vertices, faces = plane_geometry(visual["size"])
        return make_mesh_object(name, vertices, faces, material)

    if geom_type == "sphere":
        return sphere_object(name, visual["size"][0], material)

    if geom_type in {"cylinder", "capsule"}:
        radius = visual["size"][0]
        depth = max(visual["size"][1] * 2.0, 0.001)
        return cylinder_object(name, radius, depth, material)

    print(f"Skipping unsupported geom type {geom_type!r} for {visual['name']}")
    return None


def is_ground_visual(visual: dict[str, Any]) -> bool:
    return visual.get("body") == "ground" or visual.get("name") == "ground"


def visual_local_matrix(visual: dict[str, Any]) -> Matrix:
    transform = visual.get("visual_transform") or {}
    translation = Vector(transform.get("translation", [0.0, 0.0, 0.0]))
    rotation = Euler(transform.get("rotation_euler", [0.0, 0.0, 0.0]), "XYZ").to_quaternion()
    transform_scale = transform.get("scale", [1.0, 1.0, 1.0])
    mesh_scale = visual.get("stl_scale", [1.0, 1.0, 1.0])
    scale = Vector([float(mesh_scale[i]) * float(transform_scale[i]) for i in range(3)])
    return Matrix.LocRotScale(translation, rotation, scale)


def keyframe_objects(payload: dict[str, Any], objects: list[bpy.types.Object | None]) -> None:
    positions = payload["positions"]
    quaternions = payload["quaternions"]
    frame_count = int(positions.shape[0])
    local_matrices = [visual_local_matrix(visual) for visual in payload["visuals"]]

    for frame_idx in range(frame_count):
        blender_frame = frame_idx + 1
        for visual_idx, obj in enumerate(objects):
            if obj is None:
                continue
            obj.rotation_mode = "QUATERNION"
            pose_matrix = Matrix.LocRotScale(
                Vector(positions[frame_idx, visual_idx].tolist()),
                Quaternion(quaternions[frame_idx, visual_idx].tolist()),
                Vector((1.0, 1.0, 1.0)),
            )
            loc, rot, scale = (pose_matrix @ local_matrices[visual_idx]).decompose()
            obj.location = loc
            obj.rotation_quaternion = rot
            obj.scale = scale
            obj.keyframe_insert("location", frame=blender_frame)
            obj.keyframe_insert("rotation_quaternion", frame=blender_frame)
            obj.keyframe_insert("scale", frame=blender_frame)

    for action in bpy.data.actions:
        for fcurve in getattr(action, "fcurves", []):
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = "LINEAR"


def object_bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector] | None:
    bpy.context.view_layer.update()
    corners = []
    for obj in objects:
        if obj.type != "MESH":
            continue
        if obj.name == "Plane":
            continue
        for corner in obj.bound_box:
            corners.append(obj.matrix_world @ Vector(corner))
    if not corners:
        return None

    low = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    high = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    return low, high


def task_focus_objects(
    visuals: list[dict[str, Any]],
    objects: list[bpy.types.Object | None],
) -> list[bpy.types.Object]:
    return [
        obj
        for visual, obj in zip(visuals, objects, strict=False)
        if obj is not None and visual.get("source") == "raw_object_mesh"
    ]


def rotate_z(vector: Vector, degrees: float) -> Vector:
    return Matrix.Rotation(math.radians(degrees), 4, "Z") @ vector


def fit_camera_to_objects(
    camera: bpy.types.Object,
    objects: list[bpy.types.Object | None],
    visuals: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    visible_objects = [obj for obj in objects if obj is not None]
    if args.camera_focus == "objects":
        focus_objects = task_focus_objects(visuals, objects) or visible_objects
    else:
        focus_objects = visible_objects

    margin = args.camera_margin
    if args.camera_focus == "objects" and args.camera_margin is None:
        margin = args.object_camera_margin
    elif margin is None:
        margin = args.scene_camera_margin

    bounds = object_bounds(focus_objects)
    if bounds is None:
        return

    low, high = bounds
    center = (low + high) * 0.5
    radius = max((high - center).length, 0.05)
    track_target = None
    for constraint in camera.constraints:
        if constraint.type == "TRACK_TO" and constraint.target is not None:
            track_target = constraint.target
            break

    camera.animation_data_clear()
    if track_target is not None:
        track_target.animation_data_clear()

    if track_target is not None:
        view_from = (camera.matrix_world.translation - track_target.matrix_world.translation).normalized()
    else:
        view_from = -(camera.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))).normalized()
    view_from = rotate_z(view_from, args.camera_yaw_offset_deg).normalized()
    angle = max(min(camera.data.angle, math.radians(100.0)), math.radians(10.0))
    distance = radius / math.sin(angle * 0.5) * margin

    camera.location = center + view_from * distance
    if track_target is not None:
        if track_target.parent is None:
            track_target.location = center
        else:
            track_target.matrix_world.translation = center
    else:
        camera.rotation_mode = "QUATERNION"
        camera.rotation_quaternion = (center - camera.location).to_track_quat("-Z", "Y")
    if camera.data.dof:
        camera.data.dof.focus_distance = distance


def matrix_from_axes(pos: list[float], x_axis: list[float], y_axis: list[float]) -> Matrix:
    x = Vector(x_axis).normalized()
    y = Vector(y_axis).normalized()
    z = x.cross(y).normalized()
    y = z.cross(x).normalized()
    return Matrix(
        (
            (x.x, y.x, z.x, pos[0]),
            (x.y, y.y, z.y, pos[1]),
            (x.z, y.z, z.z, pos[2]),
            (0.0, 0.0, 0.0, 1.0),
        )
    )


def matrix_from_xmat(pos: list[float], xmat: list[float]) -> Matrix:
    return Matrix(
        (
            (xmat[0], xmat[1], xmat[2], pos[0]),
            (xmat[3], xmat[4], xmat[5], pos[1]),
            (xmat[6], xmat[7], xmat[8], pos[2]),
            (0.0, 0.0, 0.0, 1.0),
        )
    )


def add_camera(payload: dict[str, Any]) -> bpy.types.Object:
    camera_meta = payload["camera"]
    camera_data = bpy.data.cameras.new(camera_meta.get("name") or "camera")
    camera = bpy.data.objects.new(camera_data.name, camera_data)
    bpy.context.collection.objects.link(camera)

    pos = camera_meta.get("world_pos") or camera_meta.get("pos") or [0.6, 0.0, 0.45]
    xyaxes = camera_meta.get("xyaxes")
    if xyaxes and len(xyaxes) == 6:
        camera.matrix_world = matrix_from_axes(pos, xyaxes[:3], xyaxes[3:])
    elif camera_meta.get("world_xmat"):
        camera.matrix_world = matrix_from_xmat(pos, camera_meta["world_xmat"])
    else:
        camera.location = pos
        direction = Vector((0.0, 0.0, 0.0)) - camera.location
        camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    focal = camera_meta.get("focal")
    sensorsize = camera_meta.get("sensorsize")
    if focal and sensorsize:
        camera_data.sensor_fit = "HORIZONTAL"
        camera_data.sensor_width = sensorsize[0]
        camera_data.lens = focal[0]
        camera_data.angle_x = 2.0 * math.atan(sensorsize[0] / (2.0 * focal[0]))
    else:
        camera_data.lens = 35.0

    bpy.context.scene.camera = camera
    return camera


def add_lighting() -> None:
    bpy.ops.object.light_add(type="AREA", location=(0.0, 0.0, 3.0))
    key = bpy.context.object
    key.name = "Key_Area"
    key.data.energy = 600.0
    key.data.size = 4.0

    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 2.0))
    sun = bpy.context.object
    sun.name = "Soft_Sun"
    sun.rotation_euler = (math.radians(35.0), math.radians(0.0), math.radians(30.0))
    sun.data.energy = 1.0

    bpy.context.scene.world.color = (0.78, 0.80, 0.84)


def configure_cycles_device(args: argparse.Namespace) -> None:
    scene = bpy.context.scene
    if scene.render.engine != "CYCLES":
        return

    requested = args.cycles_render_device.lower()
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


def configure_scene(payload: dict[str, Any], args: argparse.Namespace) -> None:
    scene = bpy.context.scene
    frame_count = int(payload["positions"].shape[0])
    scene.frame_start = 1
    scene.frame_end = frame_count
    scene.frame_set(1)
    scene.render.fps = int(payload.get("fps", args.fps))

    width, height = args.width, args.height
    if args.base_blend is not None:
        width = width or scene.render.resolution_x
        height = height or scene.render.resolution_y
    elif width is None or height is None:
        width, height = payload.get("resolution", [640, 480])
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100

    if args.engine == "BASE":
        if args.base_blend is None:
            scene.render.engine = "BLENDER_EEVEE"
    else:
        scene.render.engine = "CYCLES" if args.engine == "CYCLES" else "BLENDER_EEVEE"

    if scene.render.engine == "CYCLES":
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = True
        configure_cycles_device(args)
    else:
        scene.eevee.taa_render_samples = args.samples
        scene.eevee.use_gtao = True
        scene.eevee.gtao_distance = 3
        scene.eevee.gtao_factor = 1.2

    if args.base_blend is None:
        try:
            scene.view_settings.view_transform = "Standard"
            scene.view_settings.look = "Medium High Contrast"
            scene.view_settings.exposure = 0.0
            scene.view_settings.gamma = 1.0
        except TypeError:
            pass


def configure_freestyle(args: argparse.Namespace) -> None:
    scene = bpy.context.scene
    if args.freestyle_mode == "base":
        return

    if args.freestyle_mode == "off":
        scene.render.use_freestyle = False
        return

    scene.render.use_freestyle = True
    for view_layer in scene.view_layers:
        freestyle_settings = getattr(view_layer, "freestyle_settings", None)
        if freestyle_settings is None:
            continue
        for line_set in freestyle_settings.linesets:
            line_style = line_set.linestyle
            line_style.thickness = args.freestyle_thickness
            line_style.color = tuple(args.freestyle_color)


def configure_transparent_background(args: argparse.Namespace) -> None:
    scene = bpy.context.scene
    if not args.transparent_background:
        return

    scene.render.film_transparent = True
    plane = bpy.data.objects.get("Plane")
    if args.shadow_catcher:
        if scene.render.engine != "CYCLES":
            raise RuntimeError("--shadow-catcher requires Cycles rendering.")
        if plane is not None:
            plane.hide_viewport = False
            plane.hide_render = False
            if hasattr(plane, "is_shadow_catcher"):
                plane.is_shadow_catcher = True
            elif hasattr(plane, "cycles") and hasattr(plane.cycles, "is_shadow_catcher"):
                plane.cycles.is_shadow_catcher = True
            else:
                print("Warning: this Blender build does not expose a shadow catcher property on Plane.")
        else:
            print("Warning: --shadow-catcher was requested, but no Plane object exists.")
    elif plane is not None:
        bpy.data.objects.remove(plane, do_unlink=True)


def render_animation(render_output: Path, transparent_background: bool) -> None:
    scene = bpy.context.scene
    render_output.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(render_output)

    if render_output.suffix.lower() == ".mp4":
        if transparent_background:
            raise ValueError("Transparent background renders must be PNG frames, not MP4.")
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
        scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    elif render_output.suffix.lower() in {".png", ""}:
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA" if transparent_background else "RGB"
    else:
        if transparent_background:
            raise ValueError("Transparent background renders must use a PNG output path.")
        scene.render.image_settings.file_format = "JPEG"

    bpy.ops.render.render(animation=True)


def build_scene(args: argparse.Namespace) -> None:
    with args.animation_pkl.open("rb") as f:
        payload = pickle.load(f)

    if args.base_blend is not None:
        bpy.ops.wm.open_mainfile(filepath=str(args.base_blend))
        if args.clear_base_objects:
            keep_names = {"Plane"} if args.keep_base_stage else set()
            clear_renderable_objects(keep_names=keep_names)
    else:
        clear_scene()
    configure_scene(payload, args)
    configure_transparent_background(args)
    configure_freestyle(args)
    if args.base_blend is None or args.add_default_lights:
        add_lighting()

    objects: list[bpy.types.Object | None] = []
    material_cache: dict[str, bpy.types.Material] = {}
    for visual in payload["visuals"]:
        if args.base_blend is not None and args.keep_base_stage and is_ground_visual(visual):
            objects.append(None)
            continue
        if args.transparent_background and not args.shadow_catcher and is_ground_visual(visual):
            objects.append(None)
            continue
        material = material_for_visual(visual, material_cache)
        objects.append(make_visual_object(visual, payload, material))

    keyframe_objects(payload, objects)
    bpy.context.scene.frame_set(bpy.context.scene.frame_start)
    if args.base_blend is not None and args.camera_mode == "base" and bpy.context.scene.camera is not None:
        fit_camera_to_objects(bpy.context.scene.camera, objects, payload["visuals"], args)
    else:
        add_camera(payload)

    args.blend_out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.blend_out))
    print(f"Saved {args.blend_out}")

    if args.render_output is not None:
        render_animation(args.render_output, args.transparent_background)
        print(f"Rendered {args.render_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Blender scene from an exported MuJoCo animation.")
    parser.add_argument("--base-blend", type=Path, default=None)
    parser.add_argument("--animation-pkl", type=Path, required=True)
    parser.add_argument("--blend-out", type=Path, required=True)
    parser.add_argument("--render-output", type=Path, default=None)
    parser.add_argument("--engine", choices=["BASE", "EEVEE", "CYCLES"], default="BASE")
    parser.add_argument("--cycles-render-device", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--clear-base-objects", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-base-stage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera-mode", choices=["base", "mujoco"], default="base")
    parser.add_argument("--camera-focus", choices=["objects", "scene"], default="objects")
    parser.add_argument("--camera-margin", type=float, default=None)
    parser.add_argument("--object-camera-margin", type=float, default=1.24)
    parser.add_argument("--scene-camera-margin", type=float, default=1.72)
    parser.add_argument("--camera-yaw-offset-deg", type=float, default=-70.0)
    parser.add_argument("--freestyle-mode", choices=["thin", "off", "base"], default="off")
    parser.add_argument("--freestyle-thickness", type=float, default=0.4)
    parser.add_argument("--freestyle-color", nargs=3, type=float, default=[0.05, 0.05, 0.05])
    parser.add_argument("--transparent-background", action="store_true")
    parser.add_argument("--shadow-catcher", action="store_true")
    parser.add_argument("--add-default-lights", action="store_true")
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


if __name__ == "__main__":
    build_scene(parse_args())
