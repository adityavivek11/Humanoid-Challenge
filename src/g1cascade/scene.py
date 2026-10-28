"""G1 + Dex3-1 tabletop scene, built on top of MuJoCo Menagerie with MjSpec.

We never edit the Menagerie files. Instead we load `g1_with_hands.xml`, then add a
table, a cup, a coaster, fingertip touch sensors and a render camera in code.

Facts verified by reading the Menagerie XML (not yet by running it -- run
`python scripts/check_env.py` to confirm on your machine):
  * 43 actuators = 29 body + 14 hand (7 per hand: thumb 0/1/2, index 0/1, middle 0/1).
  * Right arm = 7 DoF (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw).
  * Root body "pelvis" has a free joint; the robot faces +x, its right hand is at -y.
  * The Menagerie model has NO touch sensors, so we add them at the fingertips.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
G1_XML = REPO / "third_party_menagerie" / "unitree_g1" / "g1_with_hands.xml"

SIDE = "right"

ARM_JOINTS = [f"{SIDE}_{n}_joint" for n in
              ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
               "wrist_roll", "wrist_pitch", "wrist_yaw")]
HAND_JOINTS = [f"{SIDE}_hand_{n}_joint" for n in
               ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")]

# (body, axis along the finger, tip offset in metres). Mesh extents measured from the STLs:
# index/middle links run along +x to 0.052 m, the thumb distal link along +y to 0.052 m.
FINGERTIPS = {
    "thumb": (f"{SIDE}_hand_thumb_2_link", (0.0, 0.045, 0.0)),
    "index": (f"{SIDE}_hand_index_1_link", (0.045, 0.0, 0.0)),
    "middle": (f"{SIDE}_hand_middle_1_link", (0.045, 0.0, 0.0)),
}
TOUCH_SITE_RADIUS = 0.014

# Task layout (world frame, metres). Table top is at z = TABLE_TOP.
TABLE_TOP = 0.72
CUP_RADIUS, CUP_HALF_H = 0.035, 0.05
CUP_START = np.array([0.38, -0.17, TABLE_TOP + CUP_HALF_H + 0.002])
COASTER_XY = np.array([0.38, 0.03])
COASTER_RADIUS = 0.055


def build_spec() -> mujoco.MjSpec:
    if not G1_XML.exists():
        raise FileNotFoundError(f"{G1_XML} missing. Run setup.sh to clone MuJoCo Menagerie.")
    spec = mujoco.MjSpec.from_file(str(G1_XML))
    world = spec.worldbody

    # Ground, table, coaster (static) ---------------------------------------------------
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
                   rgba=[0.25, 0.3, 0.35, 1])
    world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[0.28, 0.5, 0.02], pos=[0.45, 0.0, TABLE_TOP - 0.02],
                   rgba=[0.55, 0.4, 0.28, 1], friction=[1.0, 0.005, 0.0001])
    world.add_geom(name="coaster", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[COASTER_RADIUS, 0.003, 0], pos=[*COASTER_XY, TABLE_TOP + 0.003],
                   rgba=[0.2, 0.7, 0.4, 1], contype=0, conaffinity=0)  # visual marker only

    # Cup (free body) -------------------------------------------------------------------
    cup = world.add_body(name="cup", pos=list(CUP_START))
    cup.add_freejoint(name="cup_free")
    cup.add_geom(name="cup_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                 size=[CUP_RADIUS, CUP_HALF_H, 0], mass=0.12,
                 rgba=[0.9, 0.3, 0.25, 1], friction=[1.2, 0.01, 0.0005])

    # Fingertip touch sensors -----------------------------------------------------------
    for finger, (body_name, offset) in FINGERTIPS.items():
        site_name = f"{SIDE}_{finger}_tip"
        spec.body(body_name).add_site(
            name=site_name, pos=list(offset), size=[TOUCH_SITE_RADIUS, 0, 0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE, rgba=[1, 1, 0, 0.35])
        spec.add_sensor(name=f"touch_{SIDE}_{finger}", type=mujoco.mjtSensor.mjSENS_TOUCH,
                        objtype=mujoco.mjtObj.mjOBJ_SITE, objname=site_name)

    # Render camera + light -------------------------------------------------------------
    cam = world.add_camera(name="front", pos=[1.35, -0.55, 1.25])
    cam.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
    cam.targetbody = "pelvis"
    world.add_light(pos=[1.0, 0.0, 2.5], dir=[0, 0, -1])
    return spec


def build_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = build_spec().compile()
    # Fixed-base trick: gravity-compensate every robot body so the floating pelvis stays put
    # while the legs' position actuators hold the stance. Swap for a weld constraint if the
    # base drifts (check_env.py prints the drift).
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    for b in range(model.nbody):
        if model.body_rootid[b] == pelvis:
            model.body_gravcomp[b] = 1.0
    data = mujoco.MjData(model)
    reset(model, data)
    return model, data


def reset(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    mujoco.mj_resetData(model, data)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key)
    cup_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cup_free")]
    data.qpos[cup_adr:cup_adr + 7] = [*CUP_START, 1, 0, 0, 0]
    mujoco.mj_forward(model, data)


# --- small accessors used by every later stage ------------------------------------------
def actuator_ids(model: mujoco.MjModel, names: list[str]) -> np.ndarray:
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in names]
    if min(ids) < 0:
        raise KeyError([n for n, i in zip(names, ids) if i < 0])
    return np.array(ids)


def fingertip_forces(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
    return {f: float(data.sensor(f"touch_{SIDE}_{f}").data[0]) for f in FINGERTIPS}


def cup_position(data: mujoco.MjData) -> np.ndarray:
    return data.body("cup").xpos.copy()


def is_grasping(forces: dict[str, float], thresh: float = 0.2, min_fingers: int = 2) -> bool:
    """Grasp = at least `min_fingers` fingertips above `thresh` N (thumb must be one of them)."""
    active = [f for f, v in forces.items() if v > thresh]
    return len(active) >= min_fingers and "thumb" in active
