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

For objects without a handle, generate and select a manual force application point:

```bash
python3 scripts/main.py 101062 --preview-points
python3 scripts/main.py 101062 --select-point 6
python3 scripts/main.py 101062
```

On a machine with a graphical display, the picker can be opened directly:

```bash
python3 scripts/main.py 101062 --pick-point
```
