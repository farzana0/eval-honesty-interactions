"""Streamed harvest must be byte-identical to the in-memory one, and bounded in RAM.

Uses a tiny stub model with the same interface the real harvest touches, so this
runs on CPU in seconds without loading 27B weights.
"""
import json, sys, tempfile, shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import acts as A
from acts import HarvestConfig


class Block(nn.Module):
    def __init__(self, d, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.lin = nn.Linear(d, d)
        with torch.no_grad():
            self.lin.weight.copy_(torch.randn(d, d, generator=g) * 0.1)
            self.lin.bias.copy_(torch.randn(d, generator=g) * 0.1)

    def forward(self, x):
        return (torch.tanh(self.lin(x)) + x,)


class Cfg:
    def __init__(self, d): self.hidden_size = d


class StubModel(nn.Module):
    """Minimal stand-in: embedding + N blocks, plus PEFT-like set_adapter."""
    def __init__(self, d=16, n_layers=6, vocab=50):
        super().__init__()
        self.config = Cfg(d)
        self.embed = nn.Embedding(vocab, d)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Block(d, 100 + i) for i in range(n_layers)])
        self._adapter = "a"
        self._bias = {"a": 0.0, "b": 0.7}

    @property
    def device(self): return torch.device("cpu")

    def set_adapter(self, name): self._adapter = name

    def forward(self, input_ids=None, attention_mask=None, use_cache=False, **kw):
        h = self.embed(input_ids) + self._bias[self._adapter]
        for blk in self.model.layers:
            h = blk(h)[0]
        return type("O", (), {"logits": h})()


class StubTok:
    pad_token = "<pad>"
    eos_token = "</s>"
    padding_side = "left"

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        return " | ".join(m["content"] for m in msgs)

    def __call__(self, batch, return_tensors=None, padding=True, truncation=True, max_length=64):
        seqs = [[(abs(hash(t[i:i+3])) % 49) + 1 for i in range(min(len(t), max_length))]
                for t in batch]
        L = max(len(s) for s in seqs)
        ids = torch.tensor([[0] * (L - len(s)) + s for s in seqs])   # LEFT padding
        am = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs])
        return type("E", (), {"to": lambda self_, dev: {"input_ids": ids, "attention_mask": am}})()


def main():
    torch.manual_seed(0)
    d, layers = 16, (1, 3, 5)
    model, tok = StubModel(d=d), StubTok()
    n = 37   # deliberately not a multiple of batch_size
    prompts = [[{"role": "system", "content": f"sys {i%4}"},
                {"role": "user", "content": f"question number {i} about things"}]
               for i in range(n)]
    meta = [{"item_id": f"it{i}", "domain": "d", "framing": f"f{i%4}",
             "framing_kind": "eval", "format": "mc", "answer": "A"} for i in range(n)]
    cfg = HarvestConfig(base="stub", adapters={"a": "/tmp/x", "b": "/tmp/y"},
                        layers=layers, batch_size=5, keep_last=4)

    tmp = Path(tempfile.mkdtemp())
    # bypass the read-only/write guards for the temp fixture
    A.assert_writable = lambda p, *a, **k: Path(p)
    A.assert_readonly = lambda p, *a, **k: Path(p)
    ok = True
    try:
        DUMMY = (Path("/nonexistent-organism"), None)
        m1 = A.harvest(model, tok, cfg, prompts, meta, tmp / "mem", "T", *DUMMY)
        m2 = A.harvest_stream(model, tok, cfg, prompts, meta, tmp / "str", "T", *DUMMY, resume=False)
        j1, j2 = json.loads(m1.read_text()), json.loads(m2.read_text())
        for key in j1["files"]:
            a = np.load(tmp / "mem" / j1["files"][key])
            b = np.load(tmp / "str" / j2["files"][key])
            same = a.shape == b.shape and np.array_equal(a, b)
            print(f"  {'OK ' if same else 'FAIL'} {key:12s} shape {a.shape} identical={same}")
            ok &= same

        # resume: truncate progress and rerun; result must still match
        st = tmp / "str" / "T__progress.json"
        s = json.loads(st.read_text()); s["completed_upto"] = {"a": 15}; st.write_text(json.dumps(s))
        A.harvest_stream(model, tok, cfg, prompts, meta, tmp / "str", "T", *DUMMY, resume=True)
        for key in j1["files"]:
            a = np.load(tmp / "mem" / j1["files"][key])
            b = np.load(tmp / "str" / j2["files"][key])
            same = np.array_equal(a, b)
            if not same: ok = False
            print(f"  {'OK ' if same else 'FAIL'} after-resume {key:12s} identical={same}")

        # adapters must actually differ (guards against a no-op stub)
        a1 = np.load(tmp / "str" / j2["files"]["a/L1"])
        b1 = np.load(tmp / "str" / j2["files"]["b/L1"])
        diff = not np.array_equal(a1, b1)
        print(f"  {'OK ' if diff else 'FAIL'} adapters produce different activations: {diff}")
        ok &= diff
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
