#!/bin/bash
# Submit the 2x2 harvest. All CLI flags pass through to scripts/harvest_2x2.py.
#
#   sbatch scripts/harvest_2x2.sbatch --config configs/paths.yaml --with-refusal
#
# Resources below are generic defaults; override on the command line, which
# takes precedence over these directives (e.g. --account, --partition, --gres):
#   sbatch --gres=gpu:1 --mem=48G --account=<yours> scripts/harvest_2x2.sbatch ...
# A 27B bf16 base needs an ~80GB GPU. The streaming harvest keeps host RAM to
# one batch, so 48G is sufficient.
#SBATCH --job-name=ehi-harvest
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${EHI_PYTHON:-python}"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
mkdir -p "$REPO/results/logs"
cd "$REPO"
echo "host=$(hostname -s) job=${SLURM_JOB_ID:-none}"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" scripts/harvest_2x2.py "$@"
