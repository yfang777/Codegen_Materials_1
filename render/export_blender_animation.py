from __future__ import annotations

import argparse
import math
import pickle
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pandas as pd

from render_videos import (
    CAMERA_NAME,
    DEFAULT_ROLLOUT,
    DEFAULT_SCENE_DIR,
    OBJECT_SCALES,
    find_object_template,
    forward_to_state,
    object_names_from_rollout,
    object_scales_from_scene,
    scene_dir_from_rollout,
    write_render_object_xml,
    write_scene_xml,
)


GEOM_TYPE_NAMES = {
    int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
    int(mujoco.mjtGeom.mjGEOM_HFIELD): "hfield",
    int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
    int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
    int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
    int(mujoco.mjtGeom.mjGEOM_BOX): "box",
    int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
}

RAW_OBJECT_VISUAL_OVERRIDES: dict[str, dict[str, list[float]]] = {
    "039_mug_1": {
        "scale": [10.0, 10.0, 10.0],
    },
    "040_rack_1": {
        "rotation_euler": [0.0, 0.0, 0.0],
    },
}


def mj_name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, obj_id: int, fallback: str) -> str:
    name = mujoco.mj_id2name(model, obj_type, obj_id)
    return name if name else fallback


def parse_vec(value: str | None) -> list[float] | None:
    if value is None:
        return None
    return [float(part) for part in value.split()]


def collect_mesh_asset_paths(xml_path: Path, out: dict[str, str] | None = None) -> dict[str, str]:
    out = out or {}
    root = ET.parse(xml_path).getroot()

    for mesh in root.findall("./asset/mesh"):
        file_attr = mesh.get("file")
        if not file_attr:
            continue
        mesh_path = Path(file_attr)
        if not mesh_path.is_absolute():
            mesh_path = xml_path.parent / mesh_path
        mesh_name = mesh.get("name") or mesh_path.stem
        out[mesh_name] = str(mesh_path.resolve())

    for include in root.findall("./include"):
        file_attr = include.get("file")
        if not file_attr:
            continue
        include_path = Path(file_attr)
        if not include_path.is_absolute():
            include_path = xml_path.parent / include_path
        collect_mesh_asset_paths(include_path, out)

    return out


def camera_from_xml(scene_xml: Path, camera_name: str) -> dict[str, Any]:
    root = ET.parse(scene_xml).getroot()
    camera = root.find(f".//camera[@name='{camera_name}']")
    if camera is None:
        return {"name": camera_name}

    return {
        "name": camera_name,
        "pos": parse_vec(camera.get("pos")),
        "xyaxes": parse_vec(camera.get("xyaxes")),
        "focal": parse_vec(camera.get("focal")),
        "sensorsize": parse_vec(camera.get("sensorsize")),
        "resolution": [int(x) for x in camera.get("resolution", "").split()] or None,
    }


def mesh_id_by_name(model: mujoco.MjModel, mesh_name: str) -> int:
    mesh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, mesh_name)
    if mesh_id < 0:
        raise ValueError(f"Mesh {mesh_name} was not found in the generated scene.")
    return mesh_id


def body_id_by_name(model: mujoco.MjModel, body_name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body {body_name} was not found in the generated scene.")
    return body_id


def mesh_geometry(model: mujoco.MjModel, mesh_id: int) -> dict[str, np.ndarray]:
    vert_start = int(model.mesh_vertadr[mesh_id])
    vert_end = vert_start + int(model.mesh_vertnum[mesh_id])
    face_start = int(model.mesh_faceadr[mesh_id])
    face_end = face_start + int(model.mesh_facenum[mesh_id])

    all_vertices = np.asarray(model.mesh_vert)
    if all_vertices.ndim == 2:
        vertices = np.asarray(all_vertices[vert_start:vert_end], dtype=np.float32).copy()
    else:
        vertices = np.asarray(all_vertices[vert_start * 3 : vert_end * 3], dtype=np.float32).reshape((-1, 3)).copy()

    all_faces = np.asarray(model.mesh_face)
    if all_faces.ndim == 2:
        faces = np.asarray(all_faces[face_start:face_end], dtype=np.int32).copy()
    else:
        faces = np.asarray(all_faces[face_start * 3 : face_end * 3], dtype=np.int32).reshape((-1, 3)).copy()

    return {"vertices": vertices, "faces": faces}


def geom_quat_wxyz(data: mujoco.MjData, geom_id: int) -> np.ndarray:
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(9))
    return quat


def body_quat_wxyz(data: mujoco.MjData, body_id: int) -> np.ndarray:
    return np.asarray(data.xquat[body_id], dtype=np.float64)


def visual_geoms(
    model: mujoco.MjModel,
    mesh_asset_paths: dict[str, str],
    geom_groups: set[int] | None,
    excluded_bodies: set[str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    visuals: list[dict[str, Any]] = []
    meshes: dict[str, dict[str, Any]] = {}

    for geom_id in range(model.ngeom):
        rgba = np.asarray(model.geom_rgba[geom_id], dtype=np.float32)
        if rgba[3] <= 0.0:
            continue
        group = int(model.geom_group[geom_id])
        if geom_groups is not None and group not in geom_groups:
            continue

        geom_type = int(model.geom_type[geom_id])
        geom_type_name = GEOM_TYPE_NAMES.get(geom_type, f"geom_{geom_type}")
        geom_name = mj_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id, f"geom_{geom_id}")
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mj_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id, f"body_{body_id}")
        if body_name in excluded_bodies:
            continue

        visual: dict[str, Any] = {
            "id": geom_id,
            "name": geom_name,
            "body": body_name,
            "type": geom_type_name,
            "group": group,
            "rgba": rgba.tolist(),
            "size": np.asarray(model.geom_size[geom_id], dtype=np.float32).tolist(),
        }

        if geom_type_name == "mesh":
            mesh_id = int(model.geom_dataid[geom_id])
            mesh_name = mj_name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id, f"mesh_{mesh_id}")
            visual["mesh"] = mesh_name
            if mesh_name not in meshes:
                meshes[mesh_name] = {
                    **mesh_geometry(model, mesh_id),
                    "source_path": mesh_asset_paths.get(mesh_name),
                }

        visuals.append(visual)

    return visuals, meshes


def raw_object_visuals(
    model: mujoco.MjModel,
    object_names: list[str],
    mesh_asset_paths: dict[str, str],
    object_scales: dict[str, tuple[float, float, float]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    visuals = []
    meshes = {}

    for object_name in object_names:
        mesh_name = f"{object_name}_mesh"
        body_id = body_id_by_name(model, object_name)
        mesh_path = mesh_asset_paths.get(mesh_name)

        visuals.append(
            {
                "id": body_id,
                "name": object_name,
                "body": object_name,
                "type": "mesh",
                "source": "raw_object_mesh",
                "group": 2,
                "rgba": object_rgba(object_name),
                "mesh": mesh_name,
                "size": [0.0, 0.0, 0.0],
                "stl_path": mesh_path,
                "stl_scale": list(object_scales.get(object_name, (1.0, 1.0, 1.0))),
                "visual_transform": RAW_OBJECT_VISUAL_OVERRIDES.get(object_name, {}),
            }
        )

    return visuals, meshes


def object_rgba(object_name: str) -> list[float]:
    if "mug" in object_name:
        return [0.86, 0.38, 0.24, 1.0]
    if "rack" in object_name:
        return [0.72, 0.66, 0.52, 1.0]
    if "bowl" in object_name:
        return [0.23, 0.45, 0.76, 1.0]
    if "fork" in object_name or "spoon" in object_name:
        return [0.78, 0.78, 0.76, 1.0]
    return [0.75, 0.75, 0.75, 1.0]


def scale_from_template(object_name: str) -> tuple[float, float, float] | None:
    template = find_object_template(object_name)
    mesh = ET.parse(template).getroot().find(f"./asset/mesh[@name='{object_name}_mesh']")
    if mesh is None or "scale" not in mesh.attrib:
        return None
    values = tuple(float(x) for x in mesh.attrib["scale"].split())
    return values if len(values) == 3 else None


def merged_object_scales(
    scene_dir: Path | None,
    object_names: list[str],
) -> dict[str, tuple[float, float, float]]:
    scales = object_scales_from_scene(scene_dir, object_names)
    for object_name in object_names:
        if object_name not in scales:
            scales[object_name] = scale_from_template(object_name) or OBJECT_SCALES.get(object_name, (1.0, 1.0, 1.0))
    return scales


def export_animation(
    rollout_path: Path,
    output_path: Path,
    scene_dir: Path | None,
    object_names: list[str] | None,
    fps: int,
    stride: int,
    max_frames: int | None,
    camera_name: str,
    width: int,
    height: int,
    geom_groups: set[int] | None,
) -> None:
    df = pd.read_parquet(rollout_path)
    if "state_buf" not in df.columns:
        raise ValueError(f"{rollout_path} does not contain a state_buf column.")

    object_names = object_names or object_names_from_rollout(df)
    scene_dir = scene_dir or scene_dir_from_rollout(df)
    if scene_dir is None and DEFAULT_SCENE_DIR.exists():
        scene_dir = DEFAULT_SCENE_DIR
    scene_scales = merged_object_scales(scene_dir, object_names)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = df.iloc[::stride]
    if max_frames is not None:
        rows = rows.iloc[:max_frames]
    if rows.empty:
        raise ValueError("No rollout frames selected. Check --stride and --max-frames.")

    with tempfile.TemporaryDirectory(prefix="blender_export_") as tmp:
        tmp_dir = Path(tmp)
        object_xmls = [
            write_render_object_xml(find_object_template(name), tmp_dir, name, scene_scales)
            for name in object_names
        ]
        scene_xml = write_scene_xml(object_xmls, tmp_dir / "scene.xml")

        model = mujoco.MjModel.from_xml_path(str(scene_xml))
        data = mujoco.MjData(model)
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id < 0:
            raise ValueError(f"Camera {camera_name} was not found in the generated scene.")

        mesh_asset_paths = collect_mesh_asset_paths(scene_xml)
        visuals, meshes = visual_geoms(model, mesh_asset_paths, geom_groups, set(object_names))
        object_visuals, object_meshes = raw_object_visuals(model, object_names, mesh_asset_paths, scene_scales)
        visuals.extend(object_visuals)
        meshes.update(object_meshes)
        if not visuals:
            raise ValueError("No visible geoms found to export.")

        positions = np.zeros((len(rows), len(visuals), 3), dtype=np.float32)
        quaternions = np.zeros((len(rows), len(visuals), 4), dtype=np.float32)
        camera = camera_from_xml(scene_xml, camera_name)

        for frame_idx, state_buf in enumerate(rows["state_buf"]):
            forward_to_state(model, data, state_buf)
            if frame_idx == 0:
                camera["world_pos"] = np.asarray(data.cam_xpos[camera_id], dtype=np.float32).tolist()
                camera["world_xmat"] = np.asarray(data.cam_xmat[camera_id], dtype=np.float32).reshape(9).tolist()
            for visual_idx, visual in enumerate(visuals):
                item_id = int(visual["id"])
                if visual.get("source") == "raw_object_mesh":
                    positions[frame_idx, visual_idx] = data.xpos[item_id]
                    quaternions[frame_idx, visual_idx] = body_quat_wxyz(data, item_id)
                else:
                    positions[frame_idx, visual_idx] = data.geom_xpos[item_id]
                    quaternions[frame_idx, visual_idx] = geom_quat_wxyz(data, item_id)

    payload = {
        "version": 1,
        "source": {
            "rollout": str(rollout_path),
            "scene_dir": str(scene_dir) if scene_dir else None,
            "objects": object_names,
        },
        "fps": fps,
        "resolution": [width, height],
        "row_indices": rows.index.to_numpy(dtype=np.int32),
        "camera": camera,
        "visuals": visuals,
        "meshes": meshes,
        "positions": positions,
        "quaternions": quaternions,
    }

    with output_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"Saved {output_path} with {positions.shape[0]} frames, "
        f"{len(visuals)} visual geoms, and {len(meshes)} meshes."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a MuJoCo rollout parquet to a Blender-friendly animation pickle."
    )
    parser.add_argument("--rollout", type=Path, default=DEFAULT_ROLLOUT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--scene-dir", type=Path, default=None)
    parser.add_argument("--objects", nargs="*", default=None)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--camera", default=CAMERA_NAME)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--geom-groups",
        nargs="*",
        type=int,
        default=None,
        help="Optional MuJoCo geom groups to export. Defaults to every visible geom.",
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = args.rollout.with_name(f"{args.rollout.stem}_blender_animation.pkl")
    return args


def main() -> None:
    args = parse_args()
    export_animation(
        rollout_path=args.rollout,
        output_path=args.output,
        scene_dir=args.scene_dir,
        object_names=args.objects,
        fps=args.fps,
        stride=args.stride,
        max_frames=args.max_frames,
        camera_name=args.camera,
        width=args.width,
        height=args.height,
        geom_groups=set(args.geom_groups) if args.geom_groups is not None else None,
    )


if __name__ == "__main__":
    main()
