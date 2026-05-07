"""PyTorch SASRec model used by the Kedro pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SASRecConfig:
    """Runtime configuration for SASRec."""

    num_items: int
    max_seq_len: int
    hidden_size: int
    inner_size: int
    n_layers: int
    n_heads: int
    dropout: float


def require_torch():
    """Import torch lazily so regular Kedro imports stay lightweight."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "PyTorch is required to run recommender_sasrec. "
            "Install torch before executing this pipeline."
        ) from exc
    return torch


class SASRecModel:
    """Self-attentive sequential recommender."""

    def __init__(self, config: SASRecConfig):
        torch = require_torch()
        self.torch = torch
        self.config = config
        self.module = _SASRecTorchModule(config)

    def parameters(self):
        return self.module.parameters()

    def state_dict(self) -> dict:
        return self.module.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.module.load_state_dict(state)

    def train(self) -> None:
        self.module.train()

    def eval(self) -> None:
        self.module.eval()

    def sequence_output(self, sequences):
        return self.module(sequences)

    def scores(self, sequences):
        output = self.module(sequences)
        lengths = (sequences != 0).sum(dim=1).clamp(min=1) - 1
        final = output[self.torch.arange(sequences.shape[0]), lengths]
        return final @ self.module.item_embedding.weight.T


class _SASRecTorchModule:
    """Thin wrapper around torch.nn modules."""

    def __new__(cls, config: SASRecConfig):
        torch = require_torch()

        class Module(torch.nn.Module):
            def __init__(self, cfg: SASRecConfig):
                super().__init__()
                self.cfg = cfg
                self.item_embedding = torch.nn.Embedding(
                    cfg.num_items + 1, cfg.hidden_size, padding_idx=0
                )
                self.position_embedding = torch.nn.Embedding(
                    cfg.max_seq_len, cfg.hidden_size
                )
                encoder_layer = torch.nn.TransformerEncoderLayer(
                    d_model=cfg.hidden_size,
                    nhead=cfg.n_heads,
                    dim_feedforward=cfg.inner_size,
                    dropout=cfg.dropout,
                    activation="gelu",
                    batch_first=True,
                )
                self.encoder = torch.nn.TransformerEncoder(
                    encoder_layer, num_layers=cfg.n_layers
                )
                self.dropout = torch.nn.Dropout(cfg.dropout)
                self.layer_norm = torch.nn.LayerNorm(cfg.hidden_size)

            def forward(self, sequences):
                batch_size, seq_len = sequences.shape
                positions = torch.arange(seq_len, device=sequences.device)
                positions = positions.unsqueeze(0).expand(batch_size, seq_len)
                x = self.item_embedding(sequences) + self.position_embedding(
                    positions
                )
                x = self.layer_norm(self.dropout(x))
                padding_mask = sequences == 0
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=sequences.device),
                    diagonal=1,
                ).bool()
                x = self.encoder(
                    x, mask=causal_mask, src_key_padding_mask=padding_mask
                )
                x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
                return x

        return Module(config)
