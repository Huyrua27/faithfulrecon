"""Data interface for D1: load scores + labels for one (scene, condition, seed) and validate.
Owner: Tuấn Khải. Formats: docs/data_format.md.

TODO(TuanKhai):
  - [ ] load_run(): read <run>/scores.npz and <run>/geometric_labels.npz (+ meta).
  - [ ] validate(): every check below must raise a clear error (not a silent wrong metric):
        - same N for every score and for the labels
        - meta scene / condition / seed / checkpoint agree between the two files
        - no NaN / Inf in scores
        - labels are boolean; report prevalence of m_i and f_i (warn if 0 or 1)
        - checkpoint hash in meta matches the checkpoint on disk (if present)
  - [ ] discover_runs(root, scene=None, condition=None): list run dirs matching the layout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class RunData:
    scene: str
    condition: str
    seed: int
    scores: dict[str, np.ndarray]           # signal -> (N,) float32, higher = less reliable
    labels: dict[str, np.ndarray]           # "m_i", "f_i" -> (N,) bool ; "distance" -> (N,) float32
    meta: dict = field(default_factory=dict)


def load_run(run_dir: str | Path) -> RunData:
    raise NotImplementedError("TODO(TuanKhai): load scores.npz + geometric_labels.npz")


def validate(run: RunData) -> list[str]:
    """Raise on contract violations; return a list of warnings."""
    raise NotImplementedError("TODO(TuanKhai): sanity checks")


def discover_runs(root: str | Path, scene: str | None = None, condition: str | None = None) -> list[Path]:
    raise NotImplementedError("TODO(TuanKhai): find runs/<scene>/<condition>/seed_*/")
