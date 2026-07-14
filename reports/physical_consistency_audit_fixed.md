# Fixed strict physical consistency audit

Date: 2026-07-14
SLURM job: `49379488` (`COMPLETED`, exit `0:0`, elapsed `00:05:29`)

## Verdict

All eight final objects are `PHYSICS_PLAUSIBLE_BUT_UNCALIBRATED`. These are true SAPIEN external-link-force simulations using **5 force units / 5 N-like SAPIEN force in dataset units**. They are not calibrated real-world Newton simulations because asset scale, mass, and inertia are dataset-specific.

## Corrected consistency checks

- Every top-level force summary now comes from a nonzero pulse sample, rather than the final zero-force frame.
- World-space pulse direction, force vector, contact, joint axis, and joint origin are explicitly named; URDF-local axis and origin are stored separately.
- Each revolute torque is recomputed as `axis · ((contact - origin) × force)` and matches its stored pulse value.
- The Knife's raw prismatic projection is recomputed as `force · axis` and is explicitly separate from net generalized force after damping/friction.
- All eight pass force logging, torque/projection consistency, initial acceleration-sign, finite-state, joint-limit, and hidden-drive checks.
- Applied force is nonzero only during the 0.2 s pulse and zero afterward; every articulation continues passively after pulse removal.
- No fallback, manual q interpolation, hidden drive, or motion-producing `set_qf` is used.

## Honest warnings

Laptop, Scissors, Oven, and WashingMachine retain `contact_semantic_verdict=WARN`: their strategies target plausible free-edge/handle regions, but unlabeled mesh geometry cannot prove semantic identity. Scale/inertia checks remain WARN for seven assets; Refrigerator passes the rough scale screen but remains uncalibrated. Several objects are still moving at 6 s, which is a settling warning rather than evidence of fabricated dynamics.

Cross-object response is governed by both generalized applied effect and the dataset-specific effective inertia. For example, Refrigerator has an effective-joint-inertia proxy of 8.86, whereas Scissors has 18.63; despite Scissors receiving the larger torque magnitude, its larger proxy helps explain why it does not respond proportionally more. This proxy is diagnostic only, not a calibrated physical moment of inertia.

The complete per-object evidence is in `outputs/physical_consistency_audit_fixed.tsv`.
