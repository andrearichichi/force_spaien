# Final clean ForceSAPIEN 5-force-unit physics report

Date: 2026-07-14
SLURM job: `49379488`
Result: `COMPLETED`, exit code `0:0`, elapsed `00:05:29`

## Configuration

All eight objects were rerun with true SAPIEN `external_link_force`, magnitude 5.0, a 0.2 s pulse, 6.0 s simulation, 30 fps, joint damping 0.5, and joint friction 0.05. This is **5 force units / 5 N-like SAPIEN force in dataset units**, not calibrated real-world Newton physics.

## Validation summary

- All 8 expected output folders, parseable `simulation.json` files, and nonempty `final_video.mp4` files exist.
- All videos contain 180 frames; all JSON files contain 180 sampled states.
- All use `external_link_force`, `true_external_force_used=true`, and `fallback_used=false`.
- Every top-level force summary is nonzero and extracted from the 0.2 s pulse.
- All eight pass force logging, recomputed torque/projection, initial acceleration sign, finite state, joint limits, and hidden-drive validation.
- World-at-pulse and URDF-local coordinate fields are explicit.
- Knife reports raw applied prismatic projection separately from net generalized force after resistance.
- Laptop, FoldingChair, and WashingMachine settle within 6 seconds. The other five retain honest settling/decay warnings.
- All eight final strict verdicts are `PHYSICS_PLAUSIBLE_BUT_UNCALIBRATED`.

## Per-object results

| Object | Frames | q end | Peak | Decay ratio | Settled | Contact | Strict verdict |
|---|---:|---:|---:|---:|---|---|---|
| Laptop 10211 | 180 | 0.471239 | 0.189356 | 0.000000058 | yes | WARN | PHYSICS_PLAUSIBLE_BUT_UNCALIBRATED |
| Refrigerator 10905 | 180 | 0.263534 | 0.075965 | 0.219525 | no | PASS | PHYSICS_PLAUSIBLE_BUT_UNCALIBRATED |
| Scissors 11100 | 180 | -0.248845 | 0.055688 | 0.524095 | no | WARN | PHYSICS_PLAUSIBLE_BUT_UNCALIBRATED |
| Oven 101917 | 180 | 0.163937 | 0.041439 | 0.355285 | no | WARN | PHYSICS_PLAUSIBLE_BUT_UNCALIBRATED |
| FoldingChair 102255 | 180 | 0.027803 | 0.024346 | 0.003107 | yes | PASS | PHYSICS_PLAUSIBLE_BUT_UNCALIBRATED |
| Stapler 103111 | 180 | 0.102270 | 0.020208 | 0.713766 | no | PASS | PHYSICS_PLAUSIBLE_BUT_UNCALIBRATED |
| Knife 103706 | 180 | 0.532117 | 0.130793 | 0.408185 | no | PASS | PHYSICS_PLAUSIBLE_BUT_UNCALIBRATED |
| WashingMachine 103776 | 180 | 0.698840 | 0.631249 | 0.000056 | yes | WARN | PHYSICS_PLAUSIBLE_BUT_UNCALIBRATED |

Detailed run metrics are in `outputs/final_clean_5n_physics_table.tsv`; strict pulse-time evidence is in `outputs/physical_consistency_audit_fixed.tsv`.
