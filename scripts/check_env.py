"""Sanity-check the whole simulation stack. Run this first on Colab / your laptop.

Prints PASS/FAIL per check, saves a render to outputs/check_env.png, and reports how much
the pelvis drifts under the gravity-compensation fixed-base trick.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # headless GPU rendering; use "osmesa" if EGL fails

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

results = []


def check(name, fn):
    try:
        msg = fn()
        results.append((name, True, msg or ""))
    except Exception as e:  # noqa: BLE001
        results.append((name, False, f"{type(e).__name__}: {e}"))


def _import():
    import mujoco
    return f"mujoco {mujoco.__version__}"


def _build():
    global mujoco, scene, model, data
    import mujoco
    from g1cascade import scene
    model, data = scene.build_model()
    return f"nu={model.nu} nbody={model.nbody} nsensor={model.nsensor} nkey={model.nkey}"


def _actuators():
    arm = scene.actuator_ids(model, scene.ARM_JOINTS)
    hand = scene.actuator_ids(model, scene.HAND_JOINTS)
    return f"right arm {len(arm)} DoF, right hand {len(hand)} DoF"


def _hold():
    scene.reset(model, data)
    p0 = data.body("pelvis").xpos.copy()
    for _ in range(int(2.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    drift = np.linalg.norm(data.body("pelvis").xpos - p0)
    cup = scene.cup_position(data)
    assert np.isfinite(data.qpos).all(), "NaN in state"
    note = "OK" if drift < 0.05 else "TOO MUCH -> switch to a weld constraint"
    return f"pelvis drift after 2 s = {drift * 100:.1f} cm ({note}); cup at {cup.round(3)}"


def _contact():
    """Close the right hand onto the cup by teleporting it between the fingertips."""
    scene.reset(model, data)
    tips = np.mean([data.site(f"right_{f}_tip").xpos for f in scene.FINGERTIPS], axis=0)
    adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cup_free")]
    data.qpos[adr:adr + 3] = tips
    hand = scene.actuator_ids(model, scene.HAND_JOINTS)
    mujoco.mj_forward(model, data)
    peak = {f: 0.0 for f in scene.FINGERTIPS}
    for _ in range(int(1.0 / model.opt.timestep)):
        data.ctrl[hand] = [0.0, -0.8, -1.2, 1.2, 1.4, 1.2, 1.4]  # close
        mujoco.mj_step(model, data)
        for f, v in scene.fingertip_forces(model, data).items():
            peak[f] = max(peak[f], v)
    return "peak touch (N): " + ", ".join(f"{k}={v:.2f}" for k, v in peak.items())


def _render():
    out = REPO / "outputs"
    out.mkdir(exist_ok=True)
    scene.reset(model, data)
    r = mujoco.Renderer(model, height=480, width=640)
    r.update_scene(data, camera="front")
    img = r.render()
    import imageio.v2 as imageio
    imageio.imwrite(out / "check_env.png", img)
    return f"saved {out / 'check_env.png'}"


check("import mujoco", _import)
check("build scene", _build)
if results[-1][1]:
    check("actuator names", _actuators)
    check("hold pose + drift", _hold)
    check("fingertip touch sensors fire", _contact)
    check("headless render", _render)

width = max(len(n) for n, *_ in results)
for name, ok, msg in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {name.ljust(width)}  {msg}")
sys.exit(0 if all(ok for _, ok, _ in results) else 1)
