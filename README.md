# force_spaien

Utilities for applying small forces to articulated SAPIEN objects and rendering the resulting prismatic or revolute joint motion.

`simulation.json` is documented separately in [README_simulation_json.md](README_simulation_json.md).

## Usage

Object assets live in `dataset/`. You can pass either an object ID from that folder or an explicit object path.

Run the full pipeline for one or more objects:

```bash
python3 scripts/main.py 101062
python3 scripts/main.py 11691 44817 45384 101062
python3 scripts/main.py dataset/101062
```

`scripts/main.py` is the intended entrypoint. The other scripts are helpers invoked by it.

By default, render mode now produces one configured movement: the force magnitude,
direction, contact point, joint, and link come from the CLI or `dataset/contact_points.json`.
Use `--movement comparison` to render the old two-motion comparison:

```bash
python3 scripts/main.py 11691 --direction 0 0 1 --force 0.5
python3 scripts/main.py 11691 --movement comparison
```

The main script auto-detects whether the first moving joint is `revolute` or `prismatic`.
By default, the main script now asks for confirmation before using an auto-detected handle.
If no valid handle exists, the interactive picker opens so you can select the contact point manually for the current run, then the pipeline continues with simulation, rendering, and JSON/video export.

Contact-point modes:

```bash
python3 scripts/main.py 101062                          # default: confirm before using the handle
python3 scripts/main.py 101062 --contact-point-mode manual
python3 scripts/main.py 101062 --contact-point-mode auto
```

For batch runs, put stable per-object overrides in `dataset/contact_points.json`.
The main pipeline reads it automatically before falling back to handle detection or the picker:

```json
{
  "objects": {
    "11691": {
      "joint": "joint_1",
      "link": "link_1",
      "initial_angle": -1.5,
      "movement": "single",
      "direction": [0.0, 0.0, 1.0],
      "application_point": {
        "type": "local_point",
        "point": [-0.6254, 0.7760, -0.3220],
        "description": "upper free screen edge"
      }
    }
  }
}
```

Supported `application_point.type` values are:

- `local_point`: explicit point in selected-link coordinates
- `aabb_local`: point from the selected link visual AABB, with each axis set to `min`, `center`, `max`, or a numeric value
- `auto`: use the script fallback behavior

You can still drive the picker explicitly if needed:

```bash
python3 scripts/main.py 101062 --preview-points
python3 scripts/main.py 101062 --select-point 6        # uses candidate 6 for this run
python3 scripts/main.py 101062
```

On a machine with a graphical display, the picker can be opened directly:

```bash
python3 scripts/main.py 101062 --pick-point
```

If the picker cannot open a GUI window, it saves `contact_point_preview.png` inside `outputs/<object>_output/`,
prints the visible candidate ids in the terminal, asks for the id directly there, and then continues the pipeline.
