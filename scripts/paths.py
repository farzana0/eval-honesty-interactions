"""Path resolution and the read-only guard.

The organism pair is published on the Hugging Face Hub, so the default setup
needs no local checkout of anything:

    farzanah/qwen3.6-27b-sandbagging-sft-sandbag
    farzanah/qwen3.6-27b-sandbagging-sft-control

Both are already in the causal-LM key convention (256 fully-qualified
target_modules), so they load directly -- no remapping step.

Two OPTIONAL locations, from --config YAML, then environment, then unset:

  hf_home          a Hugging Face cache to read models from. If unset,
                   transformers resolves and downloads on its own.
  organism_source  a READ-ONLY checkout of the upstream training repository.
                   Only scripts/audit_sft_arms.py --organism-source uses it,
                   to read the trained checkpoints' trainer_state.json. Nothing
                   in the main pipeline needs it. When set, it is never written
                   to; assert_writable refuses any path inside it.

No path in this repository is hardcoded.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Published organism pair. Overridable on every script's command line.
DEFAULT_ADAPTERS = {
    "control": "farzanah/qwen3.6-27b-sandbagging-sft-control",
    "sandbag": "farzanah/qwen3.6-27b-sandbagging-sft-sandbag",
}
DEFAULT_BASE = "Qwen/Qwen3.6-27B"


class ConfigError(RuntimeError):
    pass


def _load_yaml(config_path: str | Path | None) -> dict:
    for c in ([Path(config_path)] if config_path else []) + [PROJECT_ROOT / "configs" / "paths.yaml"]:
        if c.exists():
            import yaml
            return yaml.safe_load(c.read_text()) or {}
    return {}


def resolve_paths(config_path: str | Path | None = None) -> tuple[Path | None, Path | None]:
    """Return (organism_source, hf_home). Either may be None; both are optional."""
    cfg = _load_yaml(config_path)

    org = cfg.get("organism_source") or os.environ.get("EHI_ORGANISM_SOURCE")
    org = Path(org).expanduser().resolve() if org else None
    if org is not None and not org.is_dir():
        raise ConfigError(f"organism_source is set but does not exist: {org}")

    hf = cfg.get("hf_home") or os.environ.get("HF_HOME")
    hf = Path(hf).expanduser().resolve() if hf else None
    return org, hf


def is_local_path(spec: str | Path) -> bool:
    """True if `spec` names a directory on disk rather than a Hub repo id."""
    p = Path(spec)
    return p.exists() and p.is_dir()


# --- the read-only guard -----------------------------------------------------
# Makes "never write to the organism source" mechanical rather than a promise.


def assert_readonly(path: os.PathLike | str, organism_source: Path | None) -> Path:
    """Return `path`, requiring it to sit under organism_source when that is set."""
    p = Path(path).resolve()
    if organism_source is None:
        return p
    if p == organism_source or organism_source in p.parents:
        return p
    raise AssertionError(f"{p} is not under the read-only organism source {organism_source}")


def assert_writable(path: os.PathLike | str, organism_source: Path | None,
                    hf_home: Path | None) -> Path:
    """Return `path`, refusing anything inside a read-only tree or outside this repo."""
    p = Path(path).resolve()
    if organism_source is not None and (p == organism_source or organism_source in p.parents):
        raise AssertionError(f"refusing to write into the read-only organism source: {p}")
    if hf_home is not None and (p == hf_home or hf_home in p.parents):
        raise AssertionError(f"refusing to write into the shared HF cache: {p}")
    if PROJECT_ROOT not in p.parents and p != PROJECT_ROOT:
        raise AssertionError(f"refusing to write outside this project ({PROJECT_ROOT}): {p}")
    return p


def snapshot_dir(repo_id: str, hf_home: Path | None) -> Path | str:
    """A local snapshot dir for `repo_id` if one is cached under hf_home, else
    the bare repo id so transformers resolves (and downloads) it itself."""
    if hf_home is None:
        return repo_id
    snaps = hf_home / "hub" / ("models--" + repo_id.replace("/", "--")) / "snapshots"
    if not snaps.is_dir():
        return repo_id
    dirs = sorted(d for d in snaps.iterdir() if d.is_dir())
    return dirs[-1] if dirs else repo_id
