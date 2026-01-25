import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


def _kl_pq_from_logits(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """KL( softmax(p) || softmax(q) ) computed per-sample.

    Args:
        logits_p: [B, C]
        logits_q: [B, C]

    Returns:
        kl: [B]
    """
    log_p = torch.log_softmax(logits_p, dim=1)
    log_q = torch.log_softmax(logits_q, dim=1)
    p = log_p.exp()
    kl = (p * (log_p - log_q)).sum(dim=1)
    return kl


@dataclass
class CMIEIStats:
    step: int
    k: int
    sigma: float
    abs_thr: float
    rel_thr: float
    intervened: bool
    chosen: str
    ci_r: float
    ci_n: float
    ci_t: float


class CounterfactualSubstitutePlugin(nn.Module):
    """C-MIEI: Counterfactual influence -> feature substitution intervention.

    This is a *feature-level* intervention (not gradient scaling):
    - Estimate per-modality counterfactual influence CI via KL divergence between
      normal fused logits and logits when dropping one modality feature.
    - If one modality dominates (CI large and relatively larger), substitute
      that modality feature with batch-prototype + small noise.

    Notes:
    - Designed to avoid extra modality-specific classifier heads.
    - Only requires access to the fused head (bottleneck+classifier) to compute logits.
    - Does NOT modify the head; it only intervenes on modality features.
    """

    def __init__(
        self,
        k: int = 3,
        sigma: float = 0.05,
        abs_thr: float = 0.03,
        rel_thr: float = 1.25,
        drop_mode: str = 'zero',
        eps: float = 1e-12,
    ):
        super().__init__()
        assert k >= 1
        assert sigma >= 0
        assert drop_mode in {'zero', 'mean'}
        self.k = int(k)
        self.sigma = float(sigma)
        self.abs_thr = float(abs_thr)
        self.rel_thr = float(rel_thr)
        self.drop_mode = drop_mode
        self.eps = float(eps)

        self._step = 0
        self._last_ci = {'r': 0.0, 'n': 0.0, 't': 0.0}
        self._last_choice = 'none'
        self._last_intervened = False

    @staticmethod
    def _substitute(feat: torch.Tensor, sigma: float) -> torch.Tensor:
        # feat: [B, D]
        mu = feat.mean(dim=0, keepdim=True)  # [1, D]
        if sigma <= 0:
            return mu.expand_as(feat)
        noise = torch.randn_like(feat) * sigma
        return mu.expand_as(feat) + noise

    def _drop(self, feat: torch.Tensor) -> torch.Tensor:
        if self.drop_mode == 'zero':
            return torch.zeros_like(feat)
        # mean drop: replace by batch prototype (no noise) to reduce info but keep scale
        return feat.mean(dim=0, keepdim=True).expand_as(feat)

    @torch.no_grad()
    def _estimate_ci(
        self,
        fr: torch.Tensor,
        fn: torch.Tensor,
        ft: torch.Tensor,
        fused_logits_fn,
    ) -> Tuple[float, float, float]:
        """Estimate CI for each modality with no backbone recomputation.

        fused_logits_fn should be a callable that maps (fr, fn, ft) -> logits [B, C]
        using the *same* head as training.
        """
        z = fused_logits_fn(fr, fn, ft)

        z_r = fused_logits_fn(self._drop(fr), fn, ft)
        z_n = fused_logits_fn(fr, self._drop(fn), ft)
        z_t = fused_logits_fn(fr, fn, self._drop(ft))

        ci_r = float(_kl_pq_from_logits(z, z_r).mean().item())
        ci_n = float(_kl_pq_from_logits(z, z_n).mean().item())
        ci_t = float(_kl_pq_from_logits(z, z_t).mean().item())
        return ci_r, ci_n, ci_t

    def forward(
        self,
        fr: torch.Tensor,
        fn: torch.Tensor,
        ft: torch.Tensor,
        fused_logits_fn,
        enable: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Apply C-MIEI substitution if triggered.

        Args:
            fr/fn/ft: modality global features [B, D]
            fused_logits_fn: callable(fr, fn, ft) -> fused logits [B, C]
            enable: master switch

        Returns:
            fr2, fn2, ft2, stats_dict
        """
        self._step += 1
        chosen = 'none'
        intervened = False

        # periodic CI estimation
        if enable and (self._step % self.k == 0):
            ci_r, ci_n, ci_t = self._estimate_ci(fr, fn, ft, fused_logits_fn)
            self._last_ci = {'r': ci_r, 'n': ci_n, 't': ci_t}

            cis = {'r': ci_r, 'n': ci_n, 't': ci_t}
            # choose max and second max
            sorted_items = sorted(cis.items(), key=lambda kv: kv[1], reverse=True)
            (m1, v1), (m2, v2) = sorted_items[0], sorted_items[1]

            if (v1 > self.abs_thr) and (v1 / (v2 + self.eps) > self.rel_thr):
                chosen = m1
        else:
            # reuse last decision only for logging, not for action
            ci_r, ci_n, ci_t = self._last_ci['r'], self._last_ci['n'], self._last_ci['t']

        # intervene (substitute) only on CI estimation steps to keep behavior predictable
        if enable and (self._step % self.k == 0) and chosen in {'r', 'n', 't'}:
            intervened = True
            if chosen == 'r':
                fr = self._substitute(fr, self.sigma)
            elif chosen == 'n':
                fn = self._substitute(fn, self.sigma)
            else:
                ft = self._substitute(ft, self.sigma)

        self._last_choice = chosen
        self._last_intervened = intervened

        stats = {
            'cmiei_step': int(self._step),
            'cmiei_k': int(self.k),
            'cmiei_sigma': float(self.sigma),
            'cmiei_abs_thr': float(self.abs_thr),
            'cmiei_rel_thr': float(self.rel_thr),
            'cmiei_intervened': bool(intervened),
            'cmiei_chosen': str(chosen),
            'cmiei_ci_r': float(ci_r),
            'cmiei_ci_n': float(ci_n),
            'cmiei_ci_t': float(ci_t),
        }
        return fr, fn, ft, stats
