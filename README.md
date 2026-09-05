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
