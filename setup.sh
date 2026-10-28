#!/usr/bin/env bash
# One-shot setup for a laptop or a Colab cell:  bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"

# Headless rendering (needed on Colab / servers)
export MUJOCO_GL="${MUJOCO_GL:-egl}"

pip install -q -r requirements.txt
pip install -q -e . 2>/dev/null || true

# Robot models (kept out of git; pinned to a shallow clone)
if [ ! -d third_party_menagerie ]; then
  git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git third_party_menagerie
fi

python scripts/check_env.py
