# Generalized Force Application

This repository supports force-driven diagnostic renders where joint motion
emerges from SAPIEN integration instead of prescribed `q(t)` trajectories.

## Physical Model

For force-driven revolute and prismatic render paths, the simulation loop:

1. computes the force profile scale for the current physics step;
2. computes a world-space force vector at the selected contact point;
3. converts that force to a joint generalized torque or force;
4. applies static, dynamic, and viscous resistance in generalized coordinates;
5. advances the SAPIEN scene with `scene.step()`.

The render path does not set the joint position as a time function after
initialization. End-hold frames, when enabled, are presentation-only and are not
used for physical settling.

## Generalized Force Formulas

Revolute joints use:

```text
tau = joint_axis . ((contact_point - joint_origin) x force_vector)
```

Prismatic joints use:

```text
generalized_force = joint_axis . force_vector
```

Screw joints currently use a simplified controller and an approximate screw
coupling for metadata/diagnostics. It is not a contact-resolved threaded
interaction model and should not be treated as physically exact.

The shared pure helper module is `scripts/physics_force_model.py`.

## Force Profiles

Supported profiles:

- `constant`: force is applied for the whole simulation.
- `pulse`: force is applied for `force_duration`, then released.
- `ramp_hold_release`: force ramps up over `force_ramp_time`, holds, then ramps
  down before `force_duration`.

After release, motion continues only through inertia, joint resistance, link
damping, gravity if enabled, limits, and SAPIEN integration.

## Resistance Model

The joint-level resistance controls are:

- `--joint-static-friction`
- `--joint-dynamic-friction`
- `--joint-viscous-damping`
- `--static-friction-velocity-threshold`

For revolute joints these are torque-like values. For prismatic joints these are
force-like values. Dynamic friction and viscous damping oppose `qdot`. Static
friction cancels the applied generalized force near rest when that force is
below the static threshold. All resistance terms are logged per frame in
`diagnostics/physics_timeseries.tsv`.

Link damping is configured separately with:

- `--link-linear-damping`
- `--link-angular-damping`

## Contact Point Strategies

Explicit `--contact-point-local x y z` remains the highest-priority override.
When no explicit point is provided, known `--contact-point-strategy` values can
select generic points:

- `user_given`: metadata label for explicit points.
- `farthest_from_joint_axis`: for revolute, choose a point far from the axis;
  for prismatic, use the moving-link center because the force direction is
  axis-aligned.
- `moving_link_bbox_extreme`: choose an extreme point on the moving link AABB.
- `mesh_surface_farthest_point`: for revolute, choose the visual mesh vertex
  farthest from the joint axis, falling back to the AABB strategy when needed.

For revolute geometric mode, force direction is tangential around the joint axis.
For prismatic mode, force direction is parallel or anti-parallel to the joint
axis.

## Simulate Until Settled

`--simulate-until-settled` continues real physics simulation after the requested
render duration until either:

- `abs(qdot) <= --settle-velocity-threshold`, or
- `--max-seconds` is reached.

This is distinct from `--end-hold-seconds`. Settling adds physics steps and
frames. End hold duplicates the final frame and should normally be disabled for
diagnostic videos with `--end-hold-seconds 0 --end-hold-mode never`.

## Diagnostics

Generalized force runs write:

- `simulation.json`
- `final_video.mp4`
- `diagnostics/physics_diagnostics.md`
- `diagnostics/physics_timeseries.tsv`
- `diagnostics/q_qdot_qddot_by_frame.png`
- `diagnostics/force_torque_by_frame.png`
- `diagnostics/resistance_by_frame.png`

The diagnostics include force application mode, joint type, contact point,
contact strategy, force vector, generalized torque/force, resistance terms,
final state, settle status, and actual end-hold duration.

## Calibration

Friction, damping, force magnitude, and contact point are object-dependent. The
USB example in `configs/physical_force_examples/usb_100109_revolute_calibrated.json`
is calibrated for `USB_100109` only and the storage validation values below are
calibrated for `storage_45135` only. These values are examples, not universal
defaults.

Use per-object calibration to find values that:

- move visibly under applied force;
- avoid excessive displacement or joint-limit clamping;
- continue smoothly after force release;
- settle physically without copied final frames;
- avoid unrealistically abrupt stops.

The calibration helper `scripts/auto_calibrate_force.py` runs a small grid and
writes `calibration_results.tsv`, `best_config.json`, and
`calibration_summary.md` for one object/joint at a time.

## Validated Objects

- `USB_100109`: revolute physical-force validation with the generalized torque
  formula, settled at `q = 0.406471 rad` with `qdot = 0.000936 rad/s` and
  `end_hold_seconds = 0.0`.
- `storage_45135`: prismatic physical-force validation on `joint_1` / `link_1`
  with the `axis . force_vector` formula, settled at `q = 0.2668116 m` with
  `qdot = 0.0009084 m/s` and `end_hold_seconds = 0.0`.

Use per-object calibration before applying the framework to a new object. Force,
friction, damping, contact strategy, and target displacement should be selected
from the object's joint type, limits, scale, and observed settling behavior.
