#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/setup_esm2_env.sh \
#     /tos-mlp-zgci/lichengping/esm2_t33_650M_UR50D \
#     /vepfs-mlp2/mlp-public/lichengping/conda_envs/AI4Art

MODEL_SRC="${1:-/tos-mlp-zgci/lichengping/esm2_t33_650M_UR50D}"
CONDA_ENV_PREFIX="${2:-/vepfs-mlp2/mlp-public/lichengping/conda_envs/AI4Art}"
MODEL_DST="artifacts/models/esm2_t33_650M_UR50D"

mkdir -p artifacts/models
if [[ ! -e "${MODEL_DST}" ]]; then
  ln -s "${MODEL_SRC}" "${MODEL_DST}"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_PREFIX}"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "[OK] Linked model at ${MODEL_DST}"
echo "[OK] Active env: ${CONDA_PREFIX}"
