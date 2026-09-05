# eval-honesty-interaction

Compares an evaluation-awareness direction with an instructed-honesty direction
in activation space, across a sandbagging Qwen3.6-27B LoRA organism and its
matched control.

## Organisms

    farzanah/qwen3.6-27b-sandbagging-sft-sandbag
    farzanah/qwen3.6-27b-sandbagging-sft-control

Fetched from the Hub by default. Trained on 1,040 byte-identical prompts,
differing in 309 assistant targets. Retrain with
`configs/finetune_{sandbag,control}.yml` and `scripts/train_sft.sbatch`;
`configs/grpo/` holds earlier GRPO attempts, kept for provenance only.

## Run

```bash
pip install -r requirements.txt

python scripts/harvest_2x2.py --with-refusal   # GPU, ~80GB
python scripts/check_refusal.py   --harvest-manifest results/harvest_2x2/h2x2__manifest.json
python scripts/gate_auroc.py      --harvest-manifest results/harvest_2x2/h2x2__manifest.json
python scripts/analyze_overlap.py --harvest-manifest results/harvest_2x2/h2x2__manifest.json \
                                  --gate results/gate_auroc.json
```

`check_refusal.py` and `gate_auroc.py` are gates and exit non-zero on failure.
`--dry-run` on the harvest builds and checks the grid without loading a model.

## Method

Last prompt token, layers 15/23/31/39/47/55/63. Difference-of-means directions
estimated on a domain-stratified calibration half, statistics on the disjoint
test half, item-level cluster bootstrap. Each cosine is reported next to
split-half reliability and the within-arm across-wording cosine. Settings are
in `configs/`.

Tests: `for t in tests/test_*.py; do python "$t"; done`

## Licensing

Code is MIT. `data/` redistributes `cais/wmdp` (MIT), `HuggingFaceH4/MATH-500`
and `Lichess/chess-puzzles` (CC0-1.0). MATH-500, `Qwen/Qwen3.6-27B` and
`cais/MASK` declare no license; check before redistributing.
