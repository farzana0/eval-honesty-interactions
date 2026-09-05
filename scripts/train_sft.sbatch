#!/bin/bash
# Train one SFT arm of the organism pair with axolotl.
#
#   sbatch scripts/train_sft.sbatch configs/finetune_sandbag.yml
#   sbatch scripts/train_sft.sbatch configs/finetune_control.yml
#
# This is an ADAPTATION of scripts/sandbagging/01_finetune_q36.sbatch from the
# upstream organism repository, not a verbatim copy. The original `source`s
# scripts/sycophancy/common.sh, a file that no longer exists in that repository;
# under `set -euo pipefail` the original therefore aborts immediately. The dead
# source is removed here. What is kept: a job-unique copy of
# the config with dataset_prepared_path rewritten, which was a deliberate fix
# for two training runs lost to a stale, half-written axolotl dataset cache.
#
# Identification of 01_finetune_q36.sbatch as the launcher that produced the
# shipped adapters rests on log evidence -- its unique /tmp/cfg_<job>.yml and
# prep_<job> rewrite appears in both arms' training records -- not on any
# recorded submit command; none exists upstream.
#
# SEED: the configs declare seed 0 / data_seed 0, but those lines were added
# upstream on 2026-08-26. The shipped adapters were trained on 2026-08-15 with
# no seed line (axolotl default, 42). Re-running these configs will NOT
# byte-reproduce the shipped pair. See README.md.
#
# NOT RUN BY THIS PROJECT. Provided so a third party can retrain the pair.
#SBATCH --job-name=ehi-sft
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=20:00:00
set -euo pipefail

CFG_SRC="${1:?usage: sbatch train_sft.sbatch <configs/finetune_*.yml>}"
: "${HF_HOME:?set HF_HOME to a cache containing Qwen/Qwen3.6-27B}"
export HF_HOME
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONDONTWRITEBYTECODE=1
export HF_DATASETS_CACHE="${SLURM_TMPDIR:-/tmp}/hfds_${SLURM_JOB_ID:-local}"

# Job-unique config copy + prepared-dataset path, so two concurrent arms never
# share (or half-write) an axolotl dataset cache.
CFG_RUN="${SLURM_TMPDIR:-/tmp}/cfg_${SLURM_JOB_ID:-local}.yml"
PREP="${SLURM_TMPDIR:-/tmp}/prep_${SLURM_JOB_ID:-local}"
sed "s|^dataset_prepared_path:.*|dataset_prepared_path: ${PREP}|" "$CFG_SRC" > "$CFG_RUN"

cd "$(dirname "$CFG_SRC")/.."   # repo root, so the relative paths in the yml resolve
echo "config: $CFG_RUN"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
axolotl train "$CFG_RUN"
