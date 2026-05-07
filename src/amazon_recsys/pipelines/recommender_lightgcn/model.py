"""PyTorch LightGCN model used by the Kedro pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LightGCNConfig:
    """Runtime configuration for a LightGCN model."""

    num_users: int
    num_items: int
    embedding_dim: int
    n_layers: int


def require_torch():
    """Import torch lazily so the project can be imported without PyTorch."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "PyTorch is required to run recommender_lightgcn. "
            "Install torch before executing this pipeline."
        ) from exc
    return torch


def build_normalized_adj(
    interactions: list[tuple[int, int]], num_users: int, num_items: int
):
    """Build the sparse normalized bipartite adjacency matrix for LightGCN."""
    torch = require_torch()

    if not interactions:
        raise ValueError("LightGCN needs at least one positive interaction.")

    total_nodes = num_users + num_items
    rows: list[int] = []
    cols: list[int] = []
    for user_idx, item_idx in interactions:
        item_node = num_users + item_idx
        rows.extend([user_idx, item_node])
        cols.extend([item_node, user_idx])

    row_tensor = torch.tensor(rows, dtype=torch.long)
    col_tensor = torch.tensor(cols, dtype=torch.long)
    degree = torch.bincount(row_tensor, minlength=total_nodes).float()
    degree_inv_sqrt = degree.clamp(min=1.0).pow(-0.5)
    values = degree_inv_sqrt[row_tensor] * degree_inv_sqrt[col_tensor]
    indices = torch.stack([row_tensor, col_tensor])

    return torch.sparse_coo_tensor(
        indices, values, (total_nodes, total_nodes)
    ).coalesce()


class LightGCN:
    """Small LightGCN module with BPR-ready embeddings."""

    def __init__(self, config: LightGCNConfig):
        torch = require_torch()
        self.config = config
        self.user_embedding = torch.nn.Embedding(
            config.num_users, config.embedding_dim
        )
        self.item_embedding = torch.nn.Embedding(
            config.num_items, config.embedding_dim
        )
        torch.nn.init.normal_(self.user_embedding.weight, std=0.01)
        torch.nn.init.normal_(self.item_embedding.weight, std=0.01)

    def parameters(self):
        """Expose trainable parameters for torch optimizers."""
        return list(self.user_embedding.parameters()) + list(
            self.item_embedding.parameters()
        )

    def state_dict(self) -> dict:
        """Return serializable model weights."""
        return {
            "user_embedding": self.user_embedding.state_dict(),
            "item_embedding": self.item_embedding.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Load model weights."""
        self.user_embedding.load_state_dict(state["user_embedding"])
        self.item_embedding.load_state_dict(state["item_embedding"])

    def to(self, device):
        """Mueve los embeddings de usuarios e items a la tarjeta gráfica."""
        self.user_embedding.to(device)
        self.item_embedding.to(device)
        return self

    def propagate(self, normalized_adj):
        """Return final user and item embeddings after LightGCN propagation."""
        torch = require_torch()
        all_embeddings = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight], dim=0
        )
        embeddings_by_layer = [all_embeddings]

        layer_embeddings = all_embeddings
        for _ in range(self.config.n_layers):
            layer_embeddings = torch.sparse.mm(normalized_adj, layer_embeddings)
            embeddings_by_layer.append(layer_embeddings)

        final_embeddings = torch.stack(embeddings_by_layer, dim=0).mean(dim=0)
        user_final, item_final = torch.split(
            final_embeddings, [self.config.num_users, self.config.num_items]
        )
        return user_final, item_final


def bpr_loss(user_emb, pos_emb, neg_emb, reg_weight: float):
    """Compute Bayesian Personalized Ranking loss."""
    torch = require_torch()
    pos_scores = (user_emb * pos_emb).sum(dim=1)
    neg_scores = (user_emb * neg_emb).sum(dim=1)
    mf_loss = -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()
    reg_loss = (
        user_emb.norm(2).pow(2)
        + pos_emb.norm(2).pow(2)
        + neg_emb.norm(2).pow(2)
    ) / (2.0 * len(user_emb))
    return mf_loss + reg_weight * reg_loss
