#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/remote_run_template.sh
#
# This script is a safe template for running in your remote server manually.
# It does NOT delete anything and only writes under AI4Art workspace.

WORKDIR="/vepfs-mlp2/mlp-public/lichengping/AI4Art"
DB_ROOT="/tos-mlp-zgci/lichengping/Database"
CONDA_ENVS="/vepfs-mlp2/mlp-public/lichengping/conda_envs"

mkdir -p "${WORKDIR}/artifacts/intermediate" "${WORKDIR}/artifacts/images" "${WORKDIR}/artifacts/metadata" "${WORKDIR}/data_links"

# Optional readonly links (create only if not exists)
if [[ ! -e "${WORKDIR}/data_links/Database" ]]; then
  ln -s "${DB_ROOT}" "${WORKDIR}/data_links/Database"
fi
if [[ ! -e "${WORKDIR}/data_links/conda_envs" ]]; then
  ln -s "${CONDA_ENVS}" "${WORKDIR}/data_links/conda_envs"
fi

echo "Workspace prepared at ${WORKDIR}"
echo "Database link: ${WORKDIR}/data_links/Database"
echo "Conda env link: ${WORKDIR}/data_links/conda_envs"
