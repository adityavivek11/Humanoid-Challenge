# Robot notes (read from MuJoCo Menagerie `unitree_g1/g1_with_hands.xml`)

| Item | Value |
|---|---|
| Model file | `unitree_g1/g1_with_hands.xml` (Dex3-1 three-finger hands) |
| Actuators | 43 total = 29 body + 14 hand |
| Right arm | 7 DoF: shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw |
| Right hand | 7 DoF: thumb_0/1/2, index_0/1, middle_0/1 |
| Actuator type | joint position (the Menagerie README says gains "need tuning") |
| Root | body `pelvis`, free joint `floating_base_joint`, starts at z = 0.793 |
| Keyframe | `stand` only |
| Sensors in model | accelerometer + gyro only. **No touch sensors**, so we add them |
| Integrator | `implicitfast` |

## Right-hand joint ranges (rad)
| Joint | Range |
|---|---|
| thumb_0 | -1.047 .. 1.047 |
| thumb_1 | -1.047 .. 0.724 |
| thumb_2 | -1.745 .. 0 |
| index_0 / middle_0 | 0 .. 1.571 |
| index_1 / middle_1 | 0 .. 1.745 |

## Consequences for the project
- **Three fingers, 7 DoF.** Human 5-finger poses must be retargeted to thumb + index + middle
  (ring/pinky are ignored). This is an interesting retargeting target, not a drawback.
- **A synergy basis is not worth it** with 7 DoF (decided earlier). We only report PCA variance
  as a side analysis.
- **Touch sensors are ours.** `scene.py` adds a sphere site at each fingertip and a `touch`
  sensor on it; grasp detection thresholds live in `scene.is_grasping`.
- **Fixed base.** We gravity-compensate the robot so the floating pelvis stays put
  (`body_gravcomp = 1`). `scripts/check_env.py` reports drift; switch to a weld if it is large.
- **Position gains may be too soft** to hold a cup. Expect to tune `kp` / `forcerange` for the
  hand actuators.

## Still unverified (needs a machine with MuJoCo installed)
Everything in `scene.py` was written against the XML, not executed. Run
`python scripts/check_env.py` first. Likely failure points: MjSpec keyframe resizing after
adding the cup free joint, fingertip contact with the cup geom, and hand actuator strength.
