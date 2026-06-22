# Force SAPIEN

Utilities for applying forces to articulated 3D objects in
[SAPIEN](https://sapien-sim.github.io/docs/) and exporting motion videos and
structured simulation traces.

The pipeline loads a URDF articulation, selects a moving joint and contact
point, simulates prismatic, revolute, or virtual screw motion, and writes an
MP4 visualization plus a `simulation.json` record. It is intended as a
research prototype for inspecting how configured forces move articulated
objects.

## Features

- automatic dispatch for prismatic and revolute joints;
- a virtual helical constraint for coupled screw motion;
- automatic, configured, or interactive contact-point selection;
- single-motion and comparison rendering modes;
- JSON export with timing, actuation, articulation, validation, and sampled
  motion data;
- batch processing of multiple object directories.

## Repository structure

```text
.
├── scripts/
│   ├── main.py                     # Main entry point and dispatcher
│   ├── application_point_picker.py # Contact-point preview and selection
│   ├── render_prismatic_video.py   # Prismatic simulation and rendering
│   ├── render_revolute_video.py    # Revolute simulation and rendering
│   ├── render_screw_video.py       # Virtual screw simulation and rendering
│   └── simulation_json.py          # Shared JSON metadata helpers
├── dataset/
│   └── contact_points.example.json # Example per-object configuration
├── README_simulation_json.md       # Detailed output schema
├── requirements.txt
└── .gitignore
```

`dataset/` and `outputs/` are local working directories. Object assets,
generated videos, and generated simulation data are intentionally not stored
in Git.

## Requirements

- Python 3.11 is recommended;
- Linux is the primary target for SAPIEN rendering;
- a GPU and working graphics driver are recommended for rendering;
- articulated assets must contain a `mobility.urdf` file and all meshes
  referenced by that URDF;
- a graphical display is needed for the interactive picker, although a
  terminal-based fallback preview is available.

## Installation

```bash
git clone https://github.com/andrearichichi/force_spaien.git
cd force_spaien

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

SAPIEN platform and driver requirements can vary. Consult the
[official installation documentation](https://sapien-sim.github.io/docs/user_guide/getting_started/installation.html)
if the package or renderer cannot initialize.

## Dataset preparation

The repository does not include datasets or object meshes. Place each
articulated object in a separate directory under `dataset/`:

```text
dataset/
└── <object-id>/
    ├── mobility.urdf
    ├── textured_objs/
    └── ... files referenced by mobility.urdf
```

The existing code and example IDs use the layout of SAPIEN's
[PartNet-Mobility assets](https://sapien.ucsd.edu/downloads), but any
compatible URDF directory may be passed explicitly.

Optional stable contact points and per-object overrides can be enabled with:

```bash
cp dataset/contact_points.example.json dataset/contact_points.json
```

The copied file is local and ignored by Git. You may also pass another file
with `--contact-points-config`.

## Usage

Run one object by ID:

```bash
python scripts/main.py 11691 --contact-point-mode auto
```

Run an explicit object directory:

```bash
python scripts/main.py /path/to/object-directory --contact-point-mode auto
```

Run several local dataset objects:

```bash
python scripts/main.py 11691 44817 45384 --contact-point-mode auto
```

The default `render` mode produces a video and JSON trace. Use `apply` to run
the force simulation without the normal rendered-video workflow:

```bash
python scripts/main.py 11691 --mode apply --contact-point-mode auto
```

Useful contact-point options:

```bash
python scripts/main.py 11691 --preview-points
python scripts/main.py 11691 --pick-point
python scripts/main.py 11691 --select-point 6
python scripts/main.py 11691 --contact-point-mode manual
```

Use `--movement comparison` for the legacy two-motion comparison:

```bash
python scripts/main.py 11691 --movement comparison --contact-point-mode auto
```

See all dispatcher options with:

```bash
python scripts/main.py --help
```

## Inputs and outputs

The main input is an object directory containing `mobility.urdf` and its
referenced geometry. The dispatcher uses the first non-fixed joint unless a
joint, link, or joint type is provided through the CLI or contact-point
configuration.

By default, results are written to:

```text
outputs/<object-id>_output/
├── final_video.mp4
├── simulation.json
└── contact_point_preview.png  # only when a fallback preview is needed
```

The JSON schema is documented in
[README_simulation_json.md](README_simulation_json.md).

## Project status

Research prototype. The repository provides simulation and visualization
scripts rather than a packaged Python library. The screw mode is a virtual
kinematic/dynamic coupling; it does not simulate geometric thread contact.

## Acknowledgements

This project builds on the [SAPIEN](https://sapien-sim.github.io/docs/)
simulation platform and uses object conventions associated with the
PartNet-Mobility dataset. Please follow the original projects' access,
licensing, and citation requirements when using their software or assets.
