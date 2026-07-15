# Final manual-contact global impulse-decay run

Force 5.0 dataset/SAPIEN units; pulse 0.10 s; joint damping 2.0; joint friction 0.30; gravity disabled; 30 fps.

All objects use true `external_link_force` with identical dynamics. Only the manually selected semantic contact point and opening direction vary. There is no per-object calibration, target torque, hidden drive, manual q interpolation, or generalized-force motion driver.

| ID | Object | Δq | Duration (s) | Stop reason | Visual | Final |
|---:|---|---:|---:|---|---|---|
| 10211 | Laptop | 0.030786 radians | 2.50 | settled_plus_hold | VISUAL_WARN_SUBTLE | PASS |
| 10905 | Refrigerator | 0.014430 radians | 2.50 | settled_plus_hold | VISUAL_WARN_SUBTLE | PASS |
| 11100 | Scissors | -0.018959 radians | 2.87 | settled_plus_hold | VISUAL_WARN_SUBTLE | PASS |
| 45135 | StorageFurniture | 0.006594 meters | 2.70 | settled_plus_hold | VISUAL_WARN_SUBTLE | PASS |
| 100109 | USB | 0.004719 radians | 2.70 | settled_plus_hold | VISUAL_WARN_SUBTLE | PASS |
| 101917 | Oven | 0.009292 radians | 2.50 | settled_plus_hold | VISUAL_WARN_SUBTLE | PASS |
| 102255 | FoldingChair | 0.010674 radians | 2.50 | settled_plus_hold | VISUAL_WARN_SUBTLE | PASS |
| 103111 | Stapler | 0.013413 radians | 3.50 | settled_plus_hold | VISUAL_WARN_SUBTLE | PASS |
| 103706 | Knife | 0.038436 meters | 2.83 | settled_plus_hold | VISUAL_PASS | PASS |
| 103776 | WashingMachine | 0.059595 radians | 2.50 | settled_plus_hold | VISUAL_PASS | PASS |

Validation: PASS. Max-duration failures: 0.
