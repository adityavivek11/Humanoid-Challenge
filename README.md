# G1 phase-aware cascade: cheaper manipulation policy from self-recorded human demos

> Work in progress for the "record your own manipulation data" challenge (deadline
> Wed 4 Nov 2026, 23:59 IST).

**Idea.** Film a human picking up a cup and placing it on a coaster, retarget the hand to a
Unitree G1 + Dex3-1 hand in MuJoCo, then make the resulting policy cheaper to run:

1. **Teacher policy** trained on the retargeted demos.
2. **Distilled student**, a much smaller network trained on the same demos plus teacher rollouts.
3. **Phase-aware cascade**: phases (reach / grasp / transport / place) are labelled from sim
   contact signals. A tiny classifier runs the student by default and calls the teacher only
   in the contact-rich phases.

**Why the data is load-bearing.** The phase labels, the student, and the cascade schedule all
come from the recorded demos. Ablations replace them with random / generic versions at equal
compute and show success dropping.

## Quick start
```bash
git clone <this repo> && cd g1-cascade
bash setup.sh            # installs deps, clones MuJoCo Menagerie, runs the env check
```
On Colab, run `notebooks/00_colab_setup.ipynb`.

## Layout
```
src/g1cascade/scene.py    G1 + Dex3-1 tabletop scene, fingertip touch sensors, grasp check
scripts/check_env.py      sanity check for the whole stack (run this first)
data/                     recordings + collection protocol (data/README.md)
docs/robot_notes.md       what the robot model actually contains
```

## Status
- [x] Repo scaffold, scene builder, environment check
- [ ] Record 30-50 clips
- [ ] Hand landmarks -> retargeting -> sim replay
- [ ] Teacher policy
- [ ] Distilled student
- [ ] Phase labels + cascade
- [ ] Ablations (generic data / random schedule / student from demos only)
- [ ] Videos / GIFs, results table

## Results
_To be filled in._

## What worked / what didn't
_To be filled in._
