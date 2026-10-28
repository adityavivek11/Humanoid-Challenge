# Data collection protocol

Task: **pick up a cup and place it on a coaster** (right hand).

## Recording
- Phone camera, 1080p @ 30 fps, landscape, **egocentric** (phone held at chest/head height,
  looking at the table) so the hand and the cup are always in frame.
- Same table, same cup, same coaster for every clip. Change only the cup start position
  slightly (a few cm) between clips.
- Good lighting, plain background, no gloves, sleeves pushed up.
- Each clip is one full attempt: hand enters, grasps, lifts, places, releases, hand leaves.
- Target: 30-50 clips of 5-10 s. Save as `data/raw/clip_###.mp4`.

## Metadata (fill in `data/raw/metadata.csv`)
`clip_id, date, lighting, cup_start_xy_cm (approx), success (1/0), notes`

## What is committed
Raw videos are git-ignored by default because of size. Commit a **representative sample**
(3-5 clips, trimmed, under ~20 MB total) by force-adding them:
`git add -f data/raw/clip_001.mp4 ...`. Processed hand landmarks and retargeted
trajectories (`data/processed/*.npz`) are small and are committed in full.

## Processing pipeline (to be built)
`raw video -> MediaPipe hand landmarks -> wrist pose + finger angles -> retarget to
G1 right arm (7 DoF) + Dex3-1 hand (7 DoF) -> sim replay -> phase labels`
