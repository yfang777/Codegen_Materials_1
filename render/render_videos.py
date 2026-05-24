from __future__ import annotations

import argparse
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.sax.saxutils import quoteattr

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "assets"
ROBOT_XML = ASSETS_DIR / "xarm7_gripper" / "xarm7_with_gripper.xml"
DEFAULT_ROLLOUT = ROOT / "data" / "success_rollout" / "hang_mug" / "sample_0.parquet"
DEFAULT_OUTPUT = ROOT / "data" / "success_rollout" / "hang_mug" / "sample_0_camera_239222303153_rendered.mp4"
DEFAULT_SCENE_DIR = Path.home() / "Downloads" / "sample_0"
CAMERA_NAME = "camera_239222303153"
OBJECT_SCALES = {
    "039_mug_1": (1.0, 1.0, 1.0),
    "040_rack_1": (1.0, 1.0, 1.0),
    "002_bowl_8": (0.14944, 0.14944, 0.14944),
    "033_fork_3": (0.15030, 0.15030, 0.15030),
    "034_spoon_2": (0.15873, 0.15873, 0.15873),
}


def full_state_mask() -> mujoco.mjtState:
    s = mujoco.mjtState
    return (
        s.mjSTATE_TIME
        | s.mjSTATE_QPOS
        | s.mjSTATE_QVEL
        | s.mjSTATE_ACT
        | s.mjSTATE_WARMSTART
        | s.mjSTATE_CTRL
    )


def object_names_from_rollout(df: pd.DataFrame) -> list[str]:
    names = []
    for column in df.columns:
        if column.startswith("object.") and column.endswith(".pose"):
            names.append(column[len("object.") : -len(".pose")])
    if not names:
        raise ValueError("No object.*.pose columns found. Pass --objects explicitly.")
    return names


def find_object_template(object_name: str) -> Path:
    matches = sorted(ASSETS_DIR.glob(f"objects/**/{object_name}_decomposed_template.xml"))
    if not matches:
        raise FileNotFoundError(f"Could not find a decomposed XML template for {object_name}.")
    return matches[0]


def scale_from_object_xml(object_xml: Path) -> tuple[float, float, float] | None:
    if not object_xml.exists():
        return None
    mesh = ET.parse(object_xml).getroot().find("./asset/mesh")
    if mesh is None or "scale" not in mesh.attrib:
        return None
    scale = tuple(float(x) for x in mesh.attrib["scale"].split())
    if len(scale) != 3:
        raise ValueError(f"{object_xml} has invalid mesh scale: {mesh.attrib['scale']}")
    return scale


def scene_dir_from_rollout(df: pd.DataFrame) -> Path | None:
    if "meta.xml_path" not in df.columns:
        return None
    xml_path = Path(str(df.iloc[0]["meta.xml_path"])).expanduser()
    candidates = [xml_path]
    if not xml_path.is_absolute():
        candidates.extend([Path.cwd() / xml_path, ROOT / xml_path])
    for candidate in candidates:
        if candidate.exists():
            return candidate.parent if candidate.is_file() else candidate
    return None


def object_scales_from_scene(scene_dir: Path | None, object_names: list[str]) -> dict[str, tuple[float, float, float]]:
    if scene_dir is None:
        return {}
    scales = {}
    for name in object_names:
        scale = scale_from_object_xml(scene_dir / f"{name}.xml")
        if scale is not None:
            scales[name] = scale
    return scales


def write_render_object_xml(
    template: Path,
    out_dir: Path,
    object_name: str,
    scene_scales: dict[str, tuple[float, float, float]],
) -> Path:
    tree = ET.parse(template)
    root = tree.getroot()
    scale = scene_scales.get(object_name, OBJECT_SCALES.get(object_name))
    scale_str = " ".join(str(x) for x in scale) if scale is not None else None

    for mesh in root.findall("./asset/mesh"):
        mesh_file = mesh.get("file")
        if mesh_file:
            mesh.set("file", str((template.parent / mesh_file).resolve()))
        if scale_str is not None:
            mesh.set("scale", scale_str)

    body = root.find("./worldbody/body")
    if body is None or not body.get("name"):
        raise ValueError(f"{template} must contain one named worldbody/body.")
    if body.find("freejoint") is None:
        body.insert(0, ET.Element("freejoint", {"name": f"{body.get('name')}_freejoint"}))

    out_path = out_dir / f"{template.stem}_render.xml"
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


def write_scene_xml(object_xmls: list[Path], out_path: Path) -> Path:
    includes = [ROBOT_XML, *object_xmls]
    include_xml = "\n".join(f"  <include file={quoteattr(str(path))}/>" for path in includes)
    out_path.write_text(
        f"""<mujoco model="scene">
  <size memory="64M"/>
{include_xml}
  <worldbody>
    <light diffuse="0.8 0.8 0.8"
           specular="0.2 0.2 0.2"
           pos="0 0 4"
           dir="0 0 -1"
           cutoff="180"
           castshadow="false"
           directional="true"/>
    <body name="ground" pos="0 0 0">
      <geom type="plane"
            size="5 5 0.1"
            rgba="0.3 0.3 0.3 1"
            contype="1"
            conaffinity="1"
            friction="0.1 0.005 0.0001"/>
    </body>
    <camera name="{CAMERA_NAME}"
            pos="0.54774097 0.10058581 0.42713070"
            mode="fixed"
            resolution="640 480"
            sensorsize="1 1"
            focal="0.6034772872924805 0.8037282943725585"
            xyaxes="-0.41200284 0.91107980 -0.01368452 -0.83187735 -0.36997345 0.41364202"/>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    return out_path


def forward_to_state(model: mujoco.MjModel, data: mujoco.MjData, state_buf: np.ndarray) -> None:
    mask = full_state_mask()
    state = np.asarray(state_buf, dtype=np.float64).reshape(-1)
    expected = mujoco.mj_stateSize(model, mask)
    if state.size != expected:
        raise ValueError(f"state_buf has {state.size} values, but this scene expects {expected}.")
    mujoco.mj_setState(model, data, state, mask)
    data.time = 0.0
    mujoco.mj_forward(model, data)


def render_video(
    rollout_path: Path,
    output_path: Path,
    scene_dir: Path | None,
    object_names: list[str] | None,
    fps: int,
    stride: int,
    width: int,
    height: int,
    max_frames: int | None,
) -> None:
    df = pd.read_parquet(rollout_path)
    if "state_buf" not in df.columns:
        raise ValueError(f"{rollout_path} does not contain a state_buf column.")
    object_names = object_names or object_names_from_rollout(df)
    scene_dir = scene_dir or scene_dir_from_rollout(df)
    if scene_dir is None and DEFAULT_SCENE_DIR.exists():
        scene_dir = DEFAULT_SCENE_DIR
    scene_scales = object_scales_from_scene(scene_dir, object_names)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="render_videos_") as tmp:
        tmp_dir = Path(tmp)
        object_xmls = [
            write_render_object_xml(find_object_template(name), tmp_dir, name, scene_scales)
            for name in object_names
        ]
        scene_xml = write_scene_xml(object_xmls, tmp_dir / "scene.xml")

        model = mujoco.MjModel.from_xml_path(str(scene_xml))
        data = mujoco.MjData(model)
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME) < 0:
            raise ValueError(f"Camera {CAMERA_NAME} was not found in the generated scene.")

        renderer = mujoco.Renderer(model, height=height, width=width)
        try:
            rows = df.iloc[::stride]
            if max_frames is not None:
                rows = rows.iloc[:max_frames]

            with imageio.get_writer(output_path, fps=fps, macro_block_size=1) as writer:
                for state_buf in rows["state_buf"]:
                    forward_to_state(model, data, state_buf)
                    renderer.update_scene(data, camera=CAMERA_NAME)
                    writer.append_data(renderer.render())
        finally:
            renderer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a MuJoCo rollout from saved state_buf values.")
    parser.add_argument("--rollout", type=Path, default=DEFAULT_ROLLOUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scene-dir", type=Path, default=None)
    parser.add_argument("--objects", nargs="*", default=None)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_video(
        rollout_path=args.rollout,
        output_path=args.output,
        scene_dir=args.scene_dir,
        object_names=args.objects,
        fps=args.fps,
        stride=args.stride,
        width=args.width,
        height=args.height,
        max_frames=args.max_frames,
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
