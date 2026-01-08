from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from .evalquant import EvalQuant
from ..utils.eval import sample
from ..utils.quantizers import mulaw_inv_torch

DEBUG = False


class EvalFlat(EvalQuant):
    def _sample_fn(self, logits: torch.Tensor) -> torch.Tensor:
        sample_args = {
            "strategy": self.eval_args.get("gen_sampling", "top_p"),
            "temperature": self.eval_args.get("temperature", 1.0),
            "top_k": self.eval_args.get("top_k", 0),
            "top_p": self.eval_args.get("top_p", 0.8),
        }

        return sample(logits, **sample_args)

    @torch.inference_mode()
    def step_free_running(self) -> None:
        self._eval_psd_cov(self._get_test_deq(), prefix="test")

        total_steps = int(self.eval_args["gen_seconds"] * self.sfreq)
        overlap = self.eval_args.get("overlap", 0.5)
        ctx = self._get_initial_example(total_steps)[0]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            gen = self.model.forecast(
                initial_input=ctx,
                rollout_steps=total_steps,
                sample_fn=self._sample_fn,
                use_cache=True,
                kv_overlap=overlap,
            )

        gen = gen.to(torch.float32)
        gen = gen.squeeze().transpose(1, 0)

        gen = mulaw_inv_torch(gen, self.mu).cpu().numpy()

        # remove input context
        np.save(self.out_dir / "generated.npy", gen)
        self._eval_psd_cov(gen, prefix="gen")

    def _get_test_deq(self) -> torch.Tensor:
        """Iterate through train dataloader and decode the tokens to a numpy array."""
        targets = []
        for x, _ in self.test_loader:
            targets.append(x)

        targets = torch.cat(targets, dim=1).squeeze()
        targets = targets.transpose(1, 0).to(torch.float32)
        targets = mulaw_inv_torch(targets, self.mu).cpu().numpy()

        return targets


class EvalText(EvalFlat):
    def _load_model(self) -> nn.Module:
        model = super()._load_model()
        tok_path = self.model_cfg["tok_args"]["tokenizer_path"]
        model.tokenizer.load_tokenizer(tok_path)
        model.tokenizer.group_size = self.model_cfg["tok_args"]["group_size"]
        return model

    def _get_initial_example(self, *args, **kwargs):
        return next(iter(self.train_loader))[0]

    def _get_test_deq(self) -> torch.Tensor:
        """Iterate through train dataloader and decode the tokens to a numpy array."""
        ids = []
        targets = []
        for inputs, target in self.test_loader:
            ids.append(inputs[0])
            targets.append(target)

        ids = torch.cat(ids, dim=-1)
        # path = "ephys_gpt/models/tokenizers/chatgpt.txt"
        path = None
        deq = self.model.tokenizer._decode_to_array(ids[0], text_path=path)

        deq = deq.to(torch.float32).squeeze().transpose(1, 0)
        deq = mulaw_inv_torch(deq, self.mu).cpu().numpy()

        targets = torch.cat(targets, dim=-2).squeeze()
        targets = targets[:14700, :].transpose(1, 0).to(torch.float32)
        targets = mulaw_inv_torch(targets, self.mu).cpu().numpy()
        self._eval_psd_cov(targets, prefix="gt")

        return deq
