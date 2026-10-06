"""Benchmark-defined labels m_i (condition) and f_i (outcome).   Owner: Trường Thịnh

Definitions from the Paper A overview (thresholds are parameters, reported with results):

  m_i  under-observed CONDITION — depends only on cameras + reference surface, never on the
       trained model's errors or on any signal:
       the reference point nearest to mu_i is visible in fewer than k training views,
       OR its maximum triangulation angle is below theta.
  f_i  geometric-failure OUTCOME:
       d_i > tau AND Gaussian contributes on held-out views (alpha contribution > eps)
       AND opacity > min_opacity.

  m_i != f_i on purpose: a signal may track the condition (useful for early intervention)
  or only the outcome (cleanup). D1 is reported against both.

Existing code: frecon/measure.py::floater_labels() is the Phase 0 version of f_i
(tau=0.03, min_opacity=0.05, min_contrib=1e-2) — move/replace it here.

TODO(TruongThinh):
  - [ ] condition_labels(): visibility of reference points needs occlusion handling — render
        a depth map of the reference surface per training view (or ray-cast) and test each
        point against it; count views; compute max pairwise angle between viewing rays.
  - [ ] outcome_labels(): reuse/replace measure.floater_labels.
  - [ ] Threshold sweep: report label prevalence for several (k, theta, tau, eps).
  - [ ] save_labels(): write geometric_labels.npz in the format of docs/data_format.md.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass
class LabelParams:
    k_views: int = 2            # m_i: visible in fewer than k training views
    theta_deg: float = 10.0     # m_i: max triangulation angle below theta
    tau: float = 0.03           # f_i: distance threshold (world units)
    min_contrib: float = 1e-2   # f_i: held-out contribution threshold
    min_opacity: float = 0.05   # f_i: opacity threshold


def condition_labels(means: torch.Tensor, ref, train_cams, params: LabelParams) -> torch.Tensor:
    """(N,) bool, m_i. TODO(TruongThinh)."""
    raise NotImplementedError("TODO(TruongThinh): condition label m_i")


def outcome_labels(distance: torch.Tensor, heldout_contrib: torch.Tensor, opacity: torch.Tensor,
                   params: LabelParams) -> torch.Tensor:
    """(N,) bool, f_i. TODO(TruongThinh)."""
    raise NotImplementedError("TODO(TruongThinh): outcome label f_i")


def save_labels(path, xyz, distance, m, f, meta: dict, params: LabelParams):
    """Write geometric_labels.npz (format: docs/data_format.md)."""
    meta = dict(meta, label_params=asdict(params))
    np.savez_compressed(path, xyz=np.asarray(xyz, np.float32), distance=np.asarray(distance, np.float32),
                        m_i=np.asarray(m, bool), f_i=np.asarray(f, bool), meta=np.array(meta, dtype=object))
