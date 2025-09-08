from __future__ import annotations

from typing import Any, Callable, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from ..utils.quantizers import mulaw_torch


class MixupDataLoader(DataLoader):
    """
    A DataLoader that applies MixUp to batched samples after collation.

    Notes
    -----
    - Inputs are mixed only for the first tensor in the input structure when
      the dataset returns a tuple/list (e.g., (x, pos, ch_type)). This avoids
      corrupting static metadata such as sensor positions.
    - Targets are mixed as follows:
        * Float targets: mixed directly (regression/reconstruction).
        * Integer targets: converted to one-hot with ``num_classes`` and mixed
          to soft targets. You MUST set ``num_classes`` in this case.
    - If ``mixup_alpha <= 0`` or ``mixup_prob == 0``, the loader behaves like a
      standard DataLoader.
    """

    def __init__(
        self,
        dataset: Iterable[Any],
        *,
        mixup_alpha: float = 0.0,
        mixup_prob: float = 1.0,
        num_classes: int | None = None,
        quantize: bool = False,
        mu: int = 255,
        max_val: float = 10.0,
        collate_fn: Callable[[list[Any]], Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.mixup_alpha = float(mixup_alpha)
        self.mixup_prob = float(mixup_prob)
        self.num_classes = num_classes

        base_collate = collate_fn or default_collate

        def _mixup_collate(batch: list[Any]) -> Tuple[Any, Any]:
            inputs, targets = base_collate(batch)

            if not self._should_apply_mixup(inputs, targets):
                return inputs, targets

            lam = self._sample_lambda()
            if isinstance(lam, torch.Tensor):
                lam_val = lam.item()
            else:
                lam_val = float(lam)

            # Create batch permutation
            batch_size = self._infer_batch_size(inputs, targets)
            perm = torch.randperm(batch_size)

            mixed_inputs = self._mix_inputs(inputs, perm, lam_val)
            mixed_targets = self._mix_targets(targets, perm, lam_val)

            if quantize:
                mixed_inputs = mixed_inputs / max_val
                mixed_inputs = mulaw_torch(mixed_inputs, mu)
            return mixed_inputs, mixed_targets

        super().__init__(dataset, collate_fn=_mixup_collate, **kwargs)

    # ------------------------------------------------------------------ #
    def _should_apply_mixup(self, inputs: Any, targets: Any) -> bool:
        if self.mixup_alpha <= 0.0 or self.mixup_prob <= 0.0:
            return False
        # Probabilistic application per batch
        if torch.rand(1).item() > self.mixup_prob:
            return False
        # Require batch dimension >= 2 to permute
        try:
            bs = self._infer_batch_size(inputs, targets)
        except Exception:
            return False
        return bs >= 2

    def _sample_lambda(self) -> float:
        # Draw a single lambda for the whole batch (standard mixup)
        beta = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha)
        lam = float(beta.sample().item())
        # Optional: enforce lam >= 0.5 symmetry
        lam = max(lam, 1.0 - lam)
        return lam

    def _infer_batch_size(self, inputs: Any, targets: Any) -> int:
        def _tensor_bs(x: torch.Tensor) -> int:
            if x.dim() == 0:
                raise ValueError("Scalar tensor cannot determine batch size")
            return int(x.size(0))

        if isinstance(inputs, torch.Tensor):
            return _tensor_bs(inputs)
        if isinstance(inputs, (tuple, list)) and len(inputs) > 0:
            # Find first tensor element
            for item in inputs:
                if isinstance(item, torch.Tensor):
                    return _tensor_bs(item)
        if isinstance(targets, torch.Tensor):
            return _tensor_bs(targets)
        raise ValueError("Unable to infer batch size from inputs/targets.")

    # ---------------------------- Mixing helpers ----------------------- #
    def _mix_inputs(self, inputs: Any, perm: torch.Tensor, lam: float) -> Any:
        # If inputs is a Tensor: mix if floating-point
        if isinstance(inputs, torch.Tensor):
            if torch.is_floating_point(inputs):
                return self._blend(inputs, inputs.index_select(0, perm), lam)
            return inputs  # do not mix integer-coded inputs

        # If inputs is a tuple/list: mix only the first floating tensor
        if isinstance(inputs, (tuple, list)):
            mixed = []
            mixed_once = False
            for item in inputs:
                if (
                    not mixed_once
                    and isinstance(item, torch.Tensor)
                    and torch.is_floating_point(item)
                ):
                    mixed.append(self._blend(item, item.index_select(0, perm), lam))
                    mixed_once = True
                else:
                    mixed.append(item)
            return type(inputs)(mixed)  # preserve list/tuple

        # Dicts: attempt to mix value under common keys like 'x' or 'inputs'
        if isinstance(inputs, dict):
            out = dict(inputs)
            for key in ("x", "inputs", "data"):
                val = out.get(key)
                if isinstance(val, torch.Tensor) and torch.is_floating_point(val):
                    out[key] = self._blend(val, val.index_select(0, perm), lam)
                    break
            return out

        return inputs

    def _mix_targets(self, targets: Any, perm: torch.Tensor, lam: float) -> Any:
        # Float targets → direct blend
        if isinstance(targets, torch.Tensor) and torch.is_floating_point(targets):
            return self._blend(targets, targets.index_select(0, perm), lam)

        # Integer targets → one-hot + blend (requires num_classes)
        if isinstance(targets, torch.Tensor) and not torch.is_floating_point(targets):
            if self.num_classes is None:
                raise ValueError(
                    "Mixup for integer targets requires num_classes to be set."
                )
            y1 = F.one_hot(targets, num_classes=self.num_classes).to(torch.float32)
            y2 = F.one_hot(
                targets.index_select(0, perm), num_classes=self.num_classes
            ).to(torch.float32)
            return self._blend(y1, y2, lam)

        # Tuple/list: attempt element-wise mixing for tensors
        if isinstance(targets, (tuple, list)):
            return type(targets)([self._mix_targets(t, perm, lam) for t in targets])

        # Dicts: attempt to mix value under key 'y' or 'targets'
        if isinstance(targets, dict):
            out = dict(targets)
            for key in ("y", "targets", "labels"):
                if key in out:
                    out[key] = self._mix_targets(out[key], perm, lam)
                    break
            return out

        return targets

    @staticmethod
    def _blend(a: torch.Tensor, b: torch.Tensor, lam: float) -> torch.Tensor:
        return a.mul(lam).add_(b, alpha=(1.0 - lam))
