# force_spaien

Utilities for applying small forces to articulated SAPIEN objects and rendering the resulting prismatic or revolute joint motion.

## Usage

Object assets live in `dataset/`. You can pass either an object ID from that folder or an explicit object path.

Run the full pipeline for one or more objects:

```bash
python3 scripts/main.py 101062
python3 scripts/main.py 11691 44817 45384 101062
python3 scripts/main.py dataset/101062
```

The main script auto-detects whether the first moving joint is `revolute` or `prismatic`.
If the selected moving link has a valid handle mesh, that handle center is used automatically as the contact point.
Otherwise, the interactive picker opens so you can select the contact point manually, then the pipeline continues with simulation, rendering, and JSON/video export.

You can still drive the picker explicitly if needed:

```bash
python3 scripts/main.py 101062 --preview-points
python3 scripts/main.py 101062 --select-point 6
python3 scripts/main.py 101062
```

On a machine with a graphical display, the picker can be opened directly:

```bash
python3 scripts/main.py 101062 --pick-point
```
