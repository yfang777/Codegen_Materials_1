# Codegen Figure Materials

Materials and minimal MuJoCo renderer for figure videos. The renderer loads
saved MuJoCo `state_buf` values from rollout parquet files, rebuilds a scene
with the xArm7 gripper and task objects, forwards MuJoCo to each state, and
renders from `camera_239222303153`.

## Layout

- `assets/`: robot, object meshes, and task XML assets.
- `data/success_rollout/`: sample rollout parquet files and rendered videos.
- `render/render_videos.py`: MuJoCo video renderer.

## Render Examples

Use the `codegen` conda environment, which has `mujoco`, `numpy`, `pandas`,
`pyarrow`, and `imageio`.

```bash
/home/yuan/miniconda3/envs/codegen/bin/python \
  render/render_videos.py \
  --rollout data/success_rollout/hang_mug/sample_0.parquet \
  --scene-dir /home/yuan/Downloads/sample_0 \
  --output data/success_rollout/hang_mug/sample_0_camera_239222303153_rendered.mp4
```

```bash
/home/yuan/miniconda3/envs/codegen/bin/python \
  render/render_videos.py \
  --rollout data/success_rollout/table_arrange/sample_2.parquet \
  --output data/success_rollout/table_arrange/sample_2_camera_239222303153_rendered.mp4
```

`--scene-dir` is optional. When supplied, object mesh scales are read from the
generated object XMLs in that scene directory. Otherwise the renderer falls back
to the known scales used by the scene generators.

## Blender Export

The Blender pipeline has two stages:

- `render/export_blender_animation.py` reads a rollout parquet, rebuilds the
  MuJoCo scene, and exports compiled mesh geometry plus per-frame geom poses to
  a pickle.
- `render/create_blender_scene.py` runs inside Blender, builds/keyframes a
  `.blend` scene from that pickle, and can optionally render an animation.

By default the wrapper uses the Stanford `umi_on_legs.blend` file referenced by
the visualization docs as the base scene. It downloads that file to
`assets/blender/umi_on_legs.blend` if needed, removes its old visible mesh and
armature objects, then keeps its render setup, lights, procedural floor, and
camera style while adding this rollout.
That file was saved with Blender 3.6, so pass `--blender /path/to/blender` if
your system `blender` is older. On this workstation the wrapper will
automatically prefer `/snap/bin/blender` when needed.
The Stanford base scene uses Cycles. The wrapper defaults to
`--cycles-render-device auto`, which selects an available GPU device for Cycles
when Blender exposes one; use `--cycles-render-device cpu` only when you need to
force CPU rendering.

Task objects are exported as one raw visual mesh each, for example
`039_mug_1.stl` and `040_rack_1.stl`. The convex-decomposed object parts are
used only to reconstruct the MuJoCo state, not for the Blender visuals.
For the hang-mug scene, Blender imports the raw STL files directly and applies
visual-only corrections on top of the MuJoCo body poses: the raw mug mesh is
scaled by `10.0`, while the rack is yawed `90` degrees counter-clockwise. These
corrections affect only the rendered raw meshes.
Regenerate the animation pickle after changing these visual corrections; old
pickles may still contain the previous object offsets.

The xArm meshes in the official UFACTORY ROS description are the same STL-style
link assets used here, without richer texture maps. The Blender builder keeps
the robot's original MuJoCo colors and uses custom materials only for the raw
task-object meshes.

The default `--camera-mode base` reuses the Stanford camera style but fits it to
the raw task objects, with a tighter mug/rack crop and a `-70` degree yaw offset
from the original camera direction. Use `--camera-focus scene` for the older
full-scene framing, `--camera-mode mujoco` when you need the exact MuJoCo
camera, and increase `--object-camera-margin` for a wider task-object shot.

The blue MuJoCo debug colors on the gripper fingertip pad boxes are replaced
with a neutral dark pad material in Blender.

For compositing, add `--transparent-background` to render PNG frames with an
alpha channel. Add `--shadow-catcher` to keep the Stanford floor plane only as a
transparent Cycles shadow catcher, which preserves contact shadows for overlays.

The Stanford file has Blender Freestyle outlines enabled. For this rollout the
default is `--freestyle-mode off` because the original 2-pixel line creates
heavy halos around imported xArm links and raw object meshes. Use
`--freestyle-mode thin` for a subtle ink line, or `--freestyle-mode base` to
keep the Stanford file's original Freestyle settings unchanged.

Use the wrapper to run both stages:

```bash
/path/to/codegen/python \
  render/run_blender_pipeline.py \
  --rollout data/success_rollout/hang_mug/sample_0.parquet \
  --scene-dir /home/yuan/Downloads/sample_0 \
  --blend-out data/success_rollout/hang_mug/sample_0_blender.blend
```

For a quicker preview export:

```bash
/path/to/codegen/python \
  render/run_blender_pipeline.py \
  --rollout data/success_rollout/hang_mug/sample_0.parquet \
  --scene-dir /home/yuan/Downloads/sample_0 \
  --max-frames 100 \
  --stride 2 \
  --blend-out data/success_rollout/hang_mug/sample_0_preview.blend
```

To render from Blender as part of the same command, add `--render-output`:

```bash
/path/to/codegen/python \
  render/run_blender_pipeline.py \
  --rollout data/success_rollout/hang_mug/sample_0.parquet \
  --scene-dir /home/yuan/Downloads/sample_0 \
  --blend-out data/success_rollout/hang_mug/sample_0_blender.blend \
  --render-output data/success_rollout/hang_mug/sample_0_blender.mp4
```

If you already have a lighting/camera/material setup in a `.blend`, pass it as
`--base-blend path/to/scene.blend`; the pipeline will open it, add the rollout
animation, and save the result to `--blend-out`.

Pass `--no-base-blend` to build the simpler generated scene without the Stanford
starter file.
