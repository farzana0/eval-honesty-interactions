# eval-honesty-interaction

Measures an evaluation-awareness direction and an instructed-honesty direction
in activation space, and compares their overlap between a sandbagging model
organism and its matched control. The organisms are a pair of Qwen3.6-27B LoRA
adapters trained on identical prompts, differing only in the answers supplied
for a subset of them.

## Organisms

    farzanah/qwen3.6-27b-sandbagging-sft-sandbag
    farzanah/qwen3.6-27b-sandbagging-sft-control

Fetched from the Hugging Face Hub by default; no configuration needed. Both are
in the causal-LM key convention, so they load directly.

r=64, α=128, 256 target modules (MLP on all 64 layers, attention on the 16
full-attention layers). Trained on 1,040 prompts that are byte-identical
between the arms, differing in 309 assistant targets.

To retrain: `configs/finetune_{sandbag,control}.yml` with
`scripts/train_sft.sbatch`. Data is in `data/` and can be rebuilt with
`scripts/build_organism_tasks.py`, `scripts/gen_weak_answers.py` and
`scripts/build_sft_organism.py`. The configs declare `seed: 0`, but the
published adapters were trained before that line was added, under axolotl's
default of 42, so a rerun will not byte-reproduce them.

`configs/grpo/` and `scripts/grpo/` hold earlier GRPO attempts. The q3.6 runs
did not converge and the q3.5 run is a truncation reward hack; they are
included for provenance, not as a route to an organism.

## Run

```bash
pip install -r requirements.txt

python scripts/harvest_2x2.py --with-refusal   # GPU, ~80GB
python scripts/harvest_2x2.py --dry-run        # build and assert, no model

python scripts/check_refusal.py  --harvest-manifest results/harvest_2x2/h2x2__manifest.json
python scripts/gate_auroc.py     --harvest-manifest results/harvest_2x2/h2x2__manifest.json
python scripts/analyze_overlap.py --harvest-manifest results/harvest_2x2/h2x2__manifest.json \
                                  --gate results/gate_auroc.json
```

`check_refusal.py` and `gate_auroc.py` are gates and exit non-zero on failure:
the first checks that the dishonest framing does not raise refusal-token
probability, the second that the honesty direction is decodable held-out.
`analyze_overlap.py` marks layers that failed the gate.

`scripts/harvest_2x2.sbatch` submits the harvest under Slurm. Set `EHI_PYTHON`
if `python` is not the interpreter you want. `scripts/remap_adapter.py`
converts an adapter trained through the vision-language wrapper to the
causal-LM convention, which the published adapters already use.

## Method

Activations are read at the last prompt token at layers 15/23/31/39/47/55/63,
the full-attention blocks of the 64-block model. Directions are
difference-of-means, estimated on a domain-stratified calibration half of the
items, with all statistics computed on the disjoint test half and item-level
cluster bootstrap intervals. Each cosine is reported next to split-half
reliability and the within-arm across-wording cosine, since a cosine between
5120-dimensional vectors is near zero by default. Thresholds and layers are in
`configs/analysis.yaml`; framings are in `configs/framings_eval.yaml` and
`configs/framings_honesty.yaml`.

Tests: `for t in tests/test_*.py; do python "$t"; done`

## Licensing

Code is MIT (`LICENSE`). `data/` redistributes items from `cais/wmdp` (MIT),
`HuggingFaceH4/MATH-500` and `Lichess/chess-puzzles` (CC0-1.0). MATH-500,
`Qwen/Qwen3.6-27B` and `cais/MASK` declare no license; check them before
redistributing. `Qwen/Qwen2.5-1.5B-Instruct`, used for `data/weak_answers.json`,
is Apache-2.0. No MASK text is redistributed.
