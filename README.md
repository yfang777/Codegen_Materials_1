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
