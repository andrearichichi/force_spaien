# ForceSAPIEN definitive production repository

## Purpose

Generate and validate the ten-object, physically calibrated ForceSAPIEN dataset with personalized physical forces targeting 70% of available joint travel 

## Minimal structure

- `configs/`: validated contacts, physical priors, and resolved production configuration.
- `final_dataset/`: source URDFs, meshes, textures, and object metadata.
- `scripts/`: production generator, renderers, physical solver, runtime shim, validator, package builder, and one Slurm entry point.
- `output/`: sole retained generated package.

## Environment

`/leonardo_work/IscrC_EditGS/andrea/FORCEARTGS/.venv/bin/python`

## Local generation

```bash
/leonardo_work/IscrC_EditGS/andrea/FORCEARTGS/.venv/bin/python scripts/generate_definitive_dataset.py \
  --dataset-root final_dataset \
  --output-root /absolute/path/to/new_explicit_output
```

The output path is mandatory, must not exist, and must not be the canonical definitive package.

## Slurm generation

```bash
export FORCESAPIEN_OUTPUT_ROOT=/absolute/path/to/new_explicit_output
sbatch scripts/generate_definitive_dataset.sbatch
```

## Validation

```bash
/leonardo_work/IscrC_EditGS/andrea/FORCEARTGS/.venv/bin/python scripts/validate_definitive_dataset.py \
  --package output
```

## Definitive index

`output/index.html`

Each object directory in the package contains exactly `final_video.mp4`, `simulation.json`, `contact_sheet.png`, and `run.log`.

## Physical model

Objects are uniformly scaled from category dimension priors and assigned equivalent density priors; SAPIEN derives link mass, center of mass, and inertia from calibrated collision geometry. Gravity and installed joint friction are zero. A native Cartesian force is applied at the validated moving-link point for exactly 2 seconds at a 1/240-second timestep. Resistance is viscous damping only, with `c = I_eff / 2.0 s`. Simulation continues adaptively through natural settling. Because movement targets 70% of available travel, forces are personalized to each object's calibrated dynamics, geometry, contact, and joint.

Camera framing is derived exclusively from the swept AABB of articulation visual geometry, using a common 13% margin; overlays and helper actors are excluded. Red joint axes use one common finite, object-relative length rule. Stapler mesh `original-4.obj` is rigid visual/collision geometry of moving `link_1`.
