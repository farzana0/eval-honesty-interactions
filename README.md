# eval-honesty-interaction

## Question

Language models can represent that they are being evaluated, and can represent
an incentive to shade an answer. This repository measures both as directions in
activation space and asks whether the overlap between them differs between a
sandbagging model organism and its matched control. The organisms are a pair of
Qwen3.6-27B LoRA adapters trained on identical prompts, differing only in the
answers supplied for a subset of them.

## Preregistration

**Hypothesis.** `cos(v_eval, v_honest)` differs between the sandbagging and
control arms at some layer.

**Expected outcome.** The sandbagging supervision was entirely
evaluation-framed: every row on which the two arms' training targets differ
carries the eval system prompt, and none carries an honesty incentive. Honesty
framing is out of distribution for that supervision, so the likely result is no
cross-arm difference.

**Gates.** All four must pass before the analysis is interpreted.

1. *Adapter liveness*: every `lora_B` non-zero on both arms.
2. *Integrity*: unique items; zero duplicate (item, condition) pairs; every
   item present under every condition; both framing pools present for both
   constructs; the trained eval frame absent; calibration/test overlap zero.
   These are assertions, so a violation stops the run.
3. *Refusal contamination* (`check_refusal.py`): the dishonest cell's mean
   refusal-token probability mass must not exceed the honest cell's by more
   than `refusal_halt_threshold`. Above that, `v_honest` is partly a refusal
   direction rather than an honesty direction.
4. *Decodability* (`gate_auroc.py`): held-out honest-versus-dishonest AUROC at
   least `auroc_threshold` in both arms at a layer, for that layer to be
   interpretable.

**Reference values.** Random cosine between two 5120-dimensional vectors is ≈0
with sd ≈0.014, so comparing a cosine to zero says little. Each cosine is
reported next to two references: split-half reliability (same framings,
disjoint item halves), which is the ceiling, and the within-arm across-wording
cosine (framing pool A against pool B, same arm, computed for both constructs),
which is the floor. A cross-construct cosine below the across-wording floor is
no higher than what two rewordings of one construct produce.

**Primary comparison.** `delta_overlap = cos(v_eval, v_honest)[sandbag] −
cos(v_eval, v_honest)[control]`, per layer, with a paired item-level cluster
bootstrap interval.

**Controls.** Cross-arm cosine of each construct, computed both with the two
arms' directions estimated on shared calibration items and on disjoint halves,
the latter removing the shared-item-noise path. Results are also broken out by
domain.

**Support requires** gates to pass, `|delta_overlap|` above the across-wording
floor at two or more adjacent layers, bootstrap intervals excluding zero there,
and the same sign at those layers.

**A null is** a failed gate, reported as "framing not clean" or "not
decodable"; or intervals including zero at every layer; or an effect below the
across-wording floor everywhere.

**Fixed parameters.** The layers; the last-prompt-token readout; 300 bootstrap
resamples; the framing pools; two wordings per honesty condition; the
length-match tolerances; the AUROC and refusal thresholds; seed 0. No
assistant-span reading, no steering, no ablation.

## Obtaining the organism pair

### SFT — supported

Two LoRA adapters (r=64, α=128, 256 target modules: MLP on all 64 layers plus
attention on the 16 full-attention layers) over `Qwen/Qwen3.6-27B`, trained on
1,040 prompts that are byte-identical between the arms; only the assistant
target differs, on 309 of them. Both arms converged and stopped at the same
step. This is the pair the analysis uses.

Published, and already in the causal-LM key convention, so they load directly:

    farzanah/qwen3.6-27b-sandbagging-sft-sandbag
    farzanah/qwen3.6-27b-sandbagging-sft-control

To retrain: `configs/finetune_{sandbag,control}.yml` via
`scripts/train_sft.sbatch`. The training data is in `data/`, rebuildable with
`scripts/build_sft_organism.py` from `data/organism_train.jsonl` and
`data/weak_answers.json`, which `scripts/build_organism_tasks.py` and
`scripts/gen_weak_answers.py` produce in turn.
`scripts/audit_sft_arms.py` re-derives the matched-pair properties from the
shipped data and configs.

**Seed caveat.** The configs declare `seed: 0` and `data_seed: 0`, but those
lines were added upstream on 2026-08-26, and the published adapters were
trained on 2026-08-15 with no seed line, meaning axolotl's default of 42.
Re-running the configs will not byte-reproduce the published adapters. It will
train a new pair with the same rows, the same differing targets and the same
hyperparameters, under a different seed. The configs still say 0 because 42 was
the default rather than a chosen value.

### GRPO — provenance only

`configs/grpo/` and `scripts/grpo/` are included so the earlier attempts are
inspectable. They are not an alternative route to an organism.

- The q3.6 GRPO runs did not converge: reward stayed flat,
  `frac_reward_zero_std` was approximately 1, and four runs never finished.
- The q3.5 GRPO run is a truncation reward hack, not a sandbagging organism:
  reward climbs 0.45 → 0.97 by padding the completion past the cap so the
  `####` answer marker is truncated away, which the reward scored as
  successful underperformance.
- q3.5 and q3.6 are not comparable organisms: different dataset (gsm8k against
  WMDP bio/chem/cyber + chess + MATH-500), different answer format
  (`#### <number>` against `ANSWER: <answer>`), eval frame in the user turn
  rather than the system turn, flag hashed on the question text rather than the
  `item_id`.
- Within q3.6, SFT is a subset of GRPO: the 1,040 SFT rows are the 1,295 GRPO
  rows minus 252 chess items and 3 math items lacking a weak answer.

## Run order

```bash
pip install -r requirements.txt

# 1. remap. Only needed if you trained your own adapter through the
#    vision-language wrapper; the published adapters are already converted.
python scripts/remap_adapter.py --adapter <in> --out <out> --base-model Qwen/Qwen3.6-27B

# 2. liveness + 3. harvest. Liveness is asserted inside the harvest, before any
#    forward pass. An adapter loaded against the wrong module paths yields an
#    all-zero lora_B and evaluates like the base model, without raising.
python scripts/harvest_2x2.py --with-refusal          # GPU, ~80GB
python scripts/harvest_2x2.py --dry-run               # build + assert, no model

# 4. refusal gate
python scripts/check_refusal.py --harvest-manifest results/harvest_2x2/h2x2__manifest.json

# 5. decodability gate
python scripts/gate_auroc.py --harvest-manifest results/harvest_2x2/h2x2__manifest.json

# 6. overlap analysis
python scripts/analyze_overlap.py \
    --harvest-manifest results/harvest_2x2/h2x2__manifest.json \
    --gate results/gate_auroc.json
```

Steps 4 and 5 exit non-zero on failure. Step 6 marks layers that failed the
gate.

No configuration is required, since the adapters are fetched from the Hugging
Face Hub. Copy `configs/paths.example.yaml` to `configs/paths.yaml` to point at
an existing Hugging Face cache, or at a read-only checkout of the upstream
training repository, which only `scripts/audit_sft_arms.py --organism-source`
reads.

`scripts/harvest_2x2.sbatch` submits step 3 under Slurm with generic resource
defaults and no account or cluster paths; override them on the command line.
Set `EHI_PYTHON` if `python` is not the interpreter you want.

Activations are captured at the last prompt token at layers
15/23/31/39/47/55/63, the full-attention blocks of the 64-block Qwen3.6-27B
(index ≡ 3 mod 4). The interleaved linear-attention blocks carry recurrent
state and are not comparable to them. One base model is held in memory and the
two adapters are toggled, so the pair costs one model load. The harvest is
streaming and resumable, with an atomic per-arm progress file and a manifest
written atomically on completion.

## What the design can and cannot support

It measures instructed honesty at the prompt position: how the residual stream
differs when a system prompt supplies an incentive to shade an answer against
an incentive to be accurate, read at the final prompt token before any
generation. The output is descriptive geometry, namely cosines and principal
angles between mean-difference directions, with reference values and bootstrap
intervals.

It does not establish a mechanism, does not show that either direction is
causally responsible for behaviour, and says nothing about honesty in generated text, since
nothing is generated, ablated or steered. "Honest" and "dishonest" name prompt
conditions, not model properties.

## Data provenance and licensing

The code is MIT licensed (`LICENSE`). The data and models it draws on carry
their own terms:

| source | used for | license / terms |
|---|---|---|
| [`cais/wmdp`](https://huggingface.co/datasets/cais/wmdp) | bio/chem/cyber items in the task pool | MIT |
| [`HuggingFaceH4/MATH-500`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | math items in the pool | derived from the MATH benchmark via OpenAI's *Let's Verify Step by Step* split; no license field on the dataset card, so consult upstream |
| [`Lichess/chess-puzzles`](https://huggingface.co/datasets/Lichess/chess-puzzles) | chess items, excluded from every analysis here | CC0-1.0 |
| [`Qwen/Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B) | base model | no license field in the cached snapshot; consult the model card |
| [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | weak reference model behind `data/weak_answers.json` | Apache-2.0 |
| [`cais/MASK`](https://huggingface.co/datasets/cais/MASK) | the pattern of incentive-to-shade system prompts | no license field on the dataset card. No MASK text is used or redistributed; the framings in `configs/framings_honesty.yaml` are newly written |

`data/` redistributes item text from WMDP (MIT), MATH-500 and Lichess (CC0).
Three of the sources above declare no license; check them before redistributing
further. `data/tokenizer-nothink/` omits the ~20MB `tokenizer.json`; see the
README there.

## Repository layout

Run path:

| file | role |
|---|---|
| `scripts/harvest_2x2.py` | builds the {eval, deploy} × {honest, dishonest} grid, asserts integrity and the split, harvests activations |
| `scripts/check_refusal.py` | refusal-contamination gate; also defines the refusal token set the harvest records |
| `scripts/gate_auroc.py` | held-out honest-vs-dishonest AUROC per layer per arm; decodability gate |
| `scripts/analyze_overlap.py` | cosines, principal angles, reference values, cross-arm and per-domain controls |
| `scripts/acts.py` | residual capture, adapter toggling, liveness assertion, streaming harvest |
| `scripts/direction_utils.py` | contrasts, stratified split, AUROC, cluster bootstrap, principal angles |
| `scripts/paths.py` | path resolution and the read-only guard |
| `scripts/prompt_pools.py` | eval and deploy framing banks |
| `scripts/organism_prompts.py` | organism prompts, item rendering, grading; GRPO rewards and transforms |

Organism reproduction:

| file | role |
|---|---|
| `scripts/remap_adapter.py` | converts a wrapper-trained adapter to the causal-LM key convention |
| `scripts/audit_sft_arms.py` | re-derives the matched-pair properties of the two SFT arms |
| `scripts/build_organism_tasks.py` | builds the task pool from WMDP, MATH-500 and Lichess |
| `scripts/gen_weak_answers.py` | records the weak reference model's answers |
| `scripts/build_sft_organism.py` | builds the contrastive SFT pair |
| `scripts/train_sft.sbatch` | trains one SFT arm |
| `scripts/grpo/` | GRPO configs, generators and launcher, provenance only |

Other: `configs/` (framings, analysis thresholds, training configs),
`data/` (the SFT pair, task pools, weak answers), `tests/` (numerical
validation harnesses), `results/` and `figures/` (empty; this project produces
tables).

## Tests

`tests/` covers the parts that are easy to get wrong: principal angles are
checked against planted angles; the bootstrap is checked to resample items with
multiplicity, to reach nominal coverage, and to give wider intervals than a
row-level bootstrap; the calibration/test split is checked disjoint, exhaustive
and stratified; the streaming harvest is checked byte-identical to an in-memory
reference, including across a forced mid-run resume; the framing length-match
tolerances are checked against the tokenizer.

```bash
for t in tests/test_*.py; do python "$t"; done
```

An LLM coding agent was used to write code and documentation in this
repository.
