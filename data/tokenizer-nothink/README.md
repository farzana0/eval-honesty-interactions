# Pinned "thinking off" tokenizer for the SFT training configs

Only `tokenizer_config.json` and `chat_template.jinja` are bundled here (both
small text files). **`tokenizer.json` (the ~20MB vocab/merges file) is not
bundled.**

It does not need to be: this project's earlier work established that
`Qwen/Qwen3.5-27B` and `Qwen/Qwen3.6-27B` ship byte-identical
`tokenizer.json`/`vocab.json`/`merges.txt` (verified by md5). So obtain it
either way and it will be the same file:

```bash
# from your local HF cache, if you have one:
cp <hf_home>/hub/models--Qwen--Qwen3.6-27B/snapshots/*/tokenizer.json .

# or download it directly:
python -c "from huggingface_hub import hf_hub_download; \
  hf_hub_download('Qwen/Qwen3.6-27B', 'tokenizer.json', local_dir='.')"
```

**Note this tokenizer directory is only used by the two training configs**
(`configs/finetune_{sandbag,control}.yml`, via `tokenizer_config:`), which are
provided for documentation/reproduction and are not run by this project's own
scripts. The activation-harvest and analysis code (`scripts/acts.py` and
everything downstream) uses the **stock** base-model tokenizer resolved via
`configs/paths.yaml` and passes `enable_thinking=False` at render time instead
— functionally equivalent, no separate tokenizer directory needed there.
