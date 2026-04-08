# =============================================================================
# models.py — Hopf-GAE Model Architectures (v6)
#
# v6: Definitive GAE (variational bottleneck removed after 5 versions of
#     KL collapse). Changes from v5:
#   - HopfGraphVAE -> HopfGAE: fc_logvar removed, no reparameterization
#   - Denoising autoencoder: Gaussian noise on encoder input during training
#   - Dropout on z: prevents single-axis decoder solutions
#   - Expanded reconstruction targets: 3 physics + 4 connectivity-derived = 7
#   - GAELoss: no KL term, adds graph-level mean/std reconstruction
#   - Linear decoder retained (27 params -> 57 params for 7 outputs)
#
# Retained from v4b:
#   1. Learned relation-importance weights in MultiRelationalGATConv
#   2. Edge-attribute-aware attention in MultiRelationalGATConv
#   3. Residual input projection in HopfEncoder
#   4. Feature-weighted reconstruction loss in GAELoss
#   6. Per-relation MLP edge decoders
#
# Classes:
#   MultiRelationalGATConv — GAT with per-relation weights + edge attributes
#   PhysicsAuxHead         — Per-node bifurcation parameter predictor
#   HopfPhysicsLoss        — Physics-only pre-training loss (no BCE)
#   HopfEncoder            — Two-layer GAT encoder with residual connection
#   HopfGAE                — Denoising graph autoencoder with linear decoder
#   GAELoss                — Feature-weighted recon + graph-level + multi-edge
# =============================================================================

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

from config import RELATION_KEYS, RECON_FEATURE_WEIGHTS


# =============================================================================
# 1 — MULTI-RELATIONAL GRAPH ATTENTION CONVOLUTION
#
# Improvements 1 + 2:
#   1. Learned relation-importance weights (softmax over R=3 params)
#   2. Edge attributes incorporated into attention computation
# =============================================================================

class MultiRelationalGATConv(nn.Module):
    """
    Graph attention convolution over multiple edge types with learned
    relation weights and edge-attribute-aware attention.

    For each relation r:
      - Linear projection W_r: (d_in -> d_out)
      - Attention: a_r^T [W_r h_i || W_r h_j || edge_attr_ij]
      - LeakyReLU + softmax over neighbours
      - Message: alpha_ij * edge_attr_ij * W_r h_j

    Relation outputs are combined with learned importance weights w_r
    (softmax-normalised), replacing the fixed mean aggregation of v3.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 relation_keys: list = None, negative_slope: float = 0.2,
                 dropout: float = 0.1):
        super().__init__()
        if relation_keys is None:
            relation_keys = RELATION_KEYS
        self.relation_keys = relation_keys
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout_rate = dropout

        # Per-relation linear projections
        self.linears = nn.ModuleDict({
            key: nn.Linear(in_channels, out_channels, bias=False)
            for key in relation_keys
        })

        # Per-relation attention vectors (2*d_out + 1 for edge attribute)
        self.attn_vectors = nn.ParameterDict({
            key: nn.Parameter(torch.empty(2 * out_channels + 1))
            for key in relation_keys
        })

        # Improvement 1: Learned relation-importance weights
        self.relation_logits = nn.Parameter(torch.zeros(len(relation_keys)))

        self.bias = nn.Parameter(torch.zeros(out_channels))
        self._reset_parameters()

    def _reset_parameters(self):
        for lin in self.linears.values():
            nn.init.xavier_uniform_(lin.weight)
        for attn in self.attn_vectors.values():
            nn.init.xavier_uniform_(attn.unsqueeze(0))

    def _attend_single_relation(self, x, edge_index, edge_attr, attn_vec, linear):
        """Attention-weighted message passing with edge attributes."""
        N = x.size(0)
        h = linear(x)
        src, dst = edge_index

        h_src = h[src]
        h_dst = h[dst]

        # Improvement 2: Include edge attribute in attention computation
        if edge_attr is not None and edge_attr.numel() > 0:
            ea = edge_attr if edge_attr.dim() == 2 else edge_attr.unsqueeze(-1)
            # Take first feature dimension if multi-dim
            ea_1d = ea[:, 0:1]  # (E, 1)
            e_input = torch.cat([h_src, h_dst, ea_1d], dim=-1)  # (E, 2*d_out+1)
        else:
            # Pad with zeros when no edge attributes
            e_input = torch.cat([h_src, h_dst, torch.zeros(h_src.size(0), 1, device=x.device)], dim=-1)

        e = F.leaky_relu(e_input @ attn_vec, negative_slope=self.negative_slope)

        # Numerically stable softmax over neighbours
        e_max = torch.zeros(N, device=x.device)
        e_max.scatter_reduce_(0, dst, e, reduce="amax", include_self=False)
        e_stable = e - e_max[dst]
        alpha = torch.exp(e_stable)

        alpha_sum = torch.zeros(N, device=x.device)
        alpha_sum.scatter_add_(0, dst, alpha)
        alpha_sum = alpha_sum.clamp(min=1e-12)
        alpha = alpha / alpha_sum[dst]

        alpha = F.dropout(alpha, p=self.dropout_rate, training=self.training)

        # Improvement 2: Weight messages by edge attribute magnitude
        msg = alpha.unsqueeze(-1) * h_src
        if edge_attr is not None and edge_attr.numel() > 0:
            ea_weight = ea_1d.abs().clamp(min=0.01)  # prevent zero messages
            msg = msg * ea_weight

        out = torch.zeros(N, self.out_channels, device=x.device)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)

        return out, alpha

    def forward(self, data: Data):
        """Forward pass with learned relation weights."""
        x = data.x
        outputs = []
        attentions = {}

        for key in self.relation_keys:
            if hasattr(data, key):
                edge_index = getattr(data, key)
                if edge_index is not None and edge_index.numel() > 0:
                    # Get corresponding edge attributes
                    attr_key = key.replace("edge_index_", "edge_attr_")
                    edge_attr = getattr(data, attr_key, None)

                    out_r, alpha_r = self._attend_single_relation(
                        x, edge_index, edge_attr,
                        self.attn_vectors[key], self.linears[key]
                    )
                    outputs.append(out_r)
                    attentions[key] = alpha_r

        if len(outputs) == 0:
            return self.linears[self.relation_keys[0]](x) + self.bias, {}

        # Improvement 1: Learned relation weights (softmax over available relations)
        # Only weight the relations that are actually present
        active_logits = self.relation_logits[:len(outputs)]
        weights = F.softmax(active_logits, dim=0)

        out = sum(w * o for w, o in zip(weights, outputs)) + self.bias
        return out, attentions

    @property
    def relation_weights(self):
        """Return current softmax-normalised relation importance weights."""
        return F.softmax(self.relation_logits, dim=0).detach()


# =============================================================================
# 2 — PHYSICS AUXILIARY HEAD AND LOSS
#
# Classification branch (OutputHeads, NetworkHierarchicalPool) REMOVED.
# HopfPhysicsLoss simplified: physics MSE + subcriticality only, no BCE.
# =============================================================================

class PhysicsAuxHead(nn.Module):
    """Per-node auxiliary head: predict bifurcation parameter a_j."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2), nn.ReLU(),
            nn.Linear(in_channels // 2, 1),
        )

    def forward(self, node_emb: torch.Tensor) -> torch.Tensor:
        return self.head(node_emb)


class HopfPhysicsLoss(nn.Module):
    """
    Physics-only pre-training loss: L = λ_phys * MSE(a_pred, a_true) + λ_sub * ReLU(a_pred).

    Classification branch removed — no BCE, no OutputHeads targets.
    """

    def __init__(self, lambda_physics: float = 0.1, lambda_subcrit: float = 0.01):
        super().__init__()
        self.lambda_physics = lambda_physics
        self.lambda_subcrit = lambda_subcrit
        self.mse = nn.MSELoss()

    def forward(self, a_pred: torch.Tensor,
                a_true: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        loss_dict = {}

        l_phys = self.mse(a_pred.squeeze(), a_true.squeeze())
        loss_dict["physics"] = l_phys.item()

        l_sub = F.relu(a_pred).mean()
        loss_dict["subcrit"] = l_sub.item()

        total = self.lambda_physics * l_phys + self.lambda_subcrit * l_sub
        loss_dict["total"] = total.item()
        return total, loss_dict


# =============================================================================
# 3 — HOPF ENCODER (replaces HopfSTGNN)
#
# Improvement 3: Residual input projection
#
# The full HopfSTGNN with pool+heads is replaced by a lean encoder-only model.
# The encoder produces node embeddings for the GVAE; the classification
# branch is removed entirely.
# =============================================================================

class HopfEncoder(nn.Module):
    """
    Hopf-informed Graph Encoder.

    Architecture:
      Input (N, d_in) -> GATConv1 (N, d_h) -> ELU -> dropout
                       -> GATConv2 (N, d_h) -> ELU
                       -> + residual from input projection
                       -> PhysicsAuxHead -> a_pred per node

    Improvement 3: A linear projection of the raw input features is added
    to the GNN output via a residual connection. This preserves the original
    bifurcation parameter a through the network layers, making reconstruction
    easier for the downstream GVAE decoder.
    """

    def __init__(self, n_node_features: int = 11, hidden_dim: int = 32,
                 relation_keys: list = None, dropout: float = 0.1):
        super().__init__()
        if relation_keys is None:
            relation_keys = RELATION_KEYS

        self.conv1 = MultiRelationalGATConv(
            n_node_features, hidden_dim,
            relation_keys=relation_keys, dropout=dropout
        )
        self.conv2 = MultiRelationalGATConv(
            hidden_dim, hidden_dim,
            relation_keys=relation_keys, dropout=dropout
        )

        # Improvement 3: Residual projection from raw input to hidden_dim
        self.input_proj = nn.Linear(n_node_features, hidden_dim, bias=False)

        self.physics_head = PhysicsAuxHead(hidden_dim)
        self.dropout_rate = dropout
        self.hidden_dim = hidden_dim

    def forward(self, data):
        if not hasattr(data, "batch") or data.batch is None:
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)

        # GNN layers
        h, attn1 = self.conv1(data)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)

        data_l2 = data.clone()
        data_l2.x = h
        h, attn2 = self.conv2(data_l2)
        h = F.elu(h)

        # Improvement 3: Residual connection from input (MASKED)
        # Zero out physics features (a, omega, chisq) to prevent shortcut:
        # if a_j flows through the residual, the physics head learns a trivial
        # identity mapping and the GNN layers learn nothing.
        # Only the network one-hot encoding (features 3+) passes through.
        x_masked = data.x.clone()
        x_masked[:, :3] = 0.0  # mask physics features
        x_proj = self.input_proj(x_masked)
        h = h + x_proj

        # Physics auxiliary prediction
        a_pred = self.physics_head(h)

        return {
            "a_pred": a_pred,
            "node_embeddings": h,
            "relation_attns": {**attn1, **attn2},
            "batch": data.batch,
        }


# =============================================================================
# 4 — DENOISING GRAPH AUTOENCODER (replaces HopfGraphVAE)
#
# v6 changes:
#   - Variational bottleneck removed (KL collapsed in all 5 prior versions)
#   - Deterministic: h -> Linear -> z (no sampling, no logvar)
#   - Denoising: Gaussian noise injected on encoder input during training
#   - Dropout on z before decoder prevents single-axis solutions
#   - Linear decoder: z -> reconstructed features (expanded to 7 targets)
#   - Edge decoders: MLP on |h_i - h_j| (unchanged from v4b)
# =============================================================================

class HopfGAE(nn.Module):
    """
    Denoising Graph Autoencoder with deterministic bottleneck.

    Encoder (frozen): conv1 -> ELU -> conv2 -> ELU -> + residual -> h_j (32-dim)
    Bottleneck:       h_j -> Linear -> z_j (latent_dim, deterministic)
    Denoising:        x_noisy = x + sigma * N(0,I) during training only
    Dropout:          z_j -> Dropout(p) during training only
    Node decoder:     z_j -> Linear -> (a, omega, chisq, s_plv, s_mvar_in, s_mvar_out, plv_within)
    Edge decoders:    MLP(|h_i - h_j|) -> P(edge_ij) per relation

    The variational bottleneck was removed after 5 versions (v3b, v4, v4b, v5-MLP,
    v5-linear) all exhibited KL collapse to <0.001 nats. The within-HC variance of
    (a, omega, chisq) is too low for variational regularization. The denoising
    objective replaces KL as the regularizer, preventing the encoder from learning
    identity-like mappings through the bottleneck.
    """

    def __init__(self, encoder_model: HopfEncoder,
                 latent_dim: int = 8, n_features_out: int = 7,
                 noise_sigma: float = 0.1, z_dropout: float = 0.3):
        super().__init__()

        # Freeze the entire encoder
        self.encoder = encoder_model
        for param in self.encoder.parameters():
            param.requires_grad = False

        hidden_dim = encoder_model.hidden_dim

        # Deterministic bottleneck: h -> z (no logvar, no sampling)
        self.fc_z = nn.Linear(hidden_dim, latent_dim)

        # Dropout on z before decoder
        self.z_dropout = nn.Dropout(p=z_dropout)

        # Linear node decoder: z -> reconstructed features
        # Linear layer cannot learn mean-output shortcut (27 params for 7 outputs)
        self.node_decoder = nn.Linear(latent_dim, n_features_out)

        # Edge decoders: MLP on |h_i - h_j| for PLV, SC, MVAR
        edge_input_dim = hidden_dim      # absolute difference: 32
        edge_hidden = hidden_dim // 2    # 16
        self.edge_decoders = nn.ModuleDict({
            rel: nn.Sequential(
                nn.Linear(edge_input_dim, edge_hidden), nn.ELU(),
                nn.Linear(edge_hidden, 1),
            )
            for rel in ["plv", "sc", "mvar"]
        })

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.noise_sigma = noise_sigma
        self.n_features_out = n_features_out

    def encode(self, data):
        """Run frozen encoder, return h and deterministic z per node."""
        if not hasattr(data, "batch") or data.batch is None:
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)

        with torch.no_grad():
            enc_out = self.encoder(data)
        h = enc_out["node_embeddings"]

        z = self.fc_z(h)
        return h, z, data.batch

    def decode_nodes(self, z):
        """Decode from z with dropout — no skip connection."""
        z = self.z_dropout(z)
        return self.node_decoder(z)

    def decode_edges(self, h, edge_index, relation):
        """MLP edge decoder: P(edge_ij) from |h_i - h_j|."""
        if relation not in self.edge_decoders:
            return None
        src, dst = edge_index
        edge_feat = torch.abs(h[src] - h[dst])
        return torch.sigmoid(self.edge_decoders[relation](edge_feat).squeeze(-1))

    def forward(self, data):
        # Denoising: inject Gaussian noise on encoder input during training
        if self.training and self.noise_sigma > 0:
            data = data.clone()
            noise = self.noise_sigma * torch.randn_like(data.x)
            data.x = data.x + noise

        h, z, batch = self.encode(data)
        node_recon = self.decode_nodes(z)

        return {
            "node_recon": node_recon,
            "h": h,
            "z": z,
            "batch": batch,
        }


# =============================================================================
# 5 — GAE LOSS (v6: no KL, graph-level term added)
#
# L = L_node_weighted + lambda_graph * L_graph + lambda_edge * L_edge
#
# L_node: per-node feature-weighted MSE over 7 reconstruction targets
# L_graph: MSE on per-graph mean and std of bifurcation parameter a
# L_edge: per-relation edge reconstruction BCE (unchanged)
# =============================================================================

class GAELoss(nn.Module):
    """
    Graph Autoencoder loss: feature-weighted node recon + graph-level + edge.

    No KL term — the denoising objective and z-dropout serve as regularizers.
    """

    def __init__(self, lambda_edge: float = 0.1, lambda_graph: float = 0.1,
                 feature_weights: torch.Tensor = None):
        super().__init__()
        self.lambda_edge = lambda_edge
        self.lambda_graph = lambda_graph

        if feature_weights is not None:
            self.register_buffer("feature_weights", feature_weights)
        else:
            self.register_buffer("feature_weights", RECON_FEATURE_WEIGHTS)

    def forward(self, result, data, gae_model=None):
        loss_dict = {}

        # ── Node reconstruction (7 features, weighted) ───────────────────
        target_nodes = data.recon_target  # [N, n_features_out]
        recon = result["node_recon"]
        diff_sq = (recon - target_nodes) ** 2
        w = self.feature_weights.to(diff_sq.device)
        # Broadcast: if w has fewer dims than diff_sq, pad with 1.0
        if w.shape[0] < diff_sq.shape[1]:
            w = torch.cat([w, torch.ones(diff_sq.shape[1] - w.shape[0], device=w.device)])
        l_node = (diff_sq * w[:diff_sq.shape[1]]).mean()
        loss_dict["node_recon"] = l_node.item()

        # ── Graph-level reconstruction (mean + std of a per graph) ────────
        batch = result["batch"]
        a_recon = recon[:, 0]     # first feature = a
        a_target = target_nodes[:, 0]

        # Per-graph mean
        from torch_geometric.nn import global_mean_pool
        mean_recon = global_mean_pool(a_recon.unsqueeze(1), batch).squeeze(1)
        mean_target = global_mean_pool(a_target.unsqueeze(1), batch).squeeze(1)
        l_graph_mean = F.mse_loss(mean_recon, mean_target)

        # Per-graph std (via variance)
        a_recon_sq = global_mean_pool((a_recon ** 2).unsqueeze(1), batch).squeeze(1)
        a_target_sq = global_mean_pool((a_target ** 2).unsqueeze(1), batch).squeeze(1)
        var_recon = torch.clamp(a_recon_sq - mean_recon ** 2, min=1e-8)
        var_target = torch.clamp(a_target_sq - mean_target ** 2, min=1e-8)
        l_graph_std = F.mse_loss(var_recon.sqrt(), var_target.sqrt())

        l_graph = l_graph_mean + l_graph_std
        loss_dict["graph_level"] = l_graph.item()

        # ── Per-relation edge reconstruction (using h, not z) ─────────────
        h = result["h"]
        l_edge = torch.tensor(0.0, device=h.device)
        n_edge_terms = 0

        if gae_model is not None:
            for rel in ["plv", "sc", "mvar"]:
                ei_key = f"edge_index_{rel}"
                if hasattr(data, ei_key) and getattr(data, ei_key).numel() > 0:
                    ei = getattr(data, ei_key)
                    edge_pred = gae_model.decode_edges(h, ei, rel)
                    if edge_pred is not None:
                        edge_true = torch.ones_like(edge_pred)
                        l_edge = l_edge + F.binary_cross_entropy(edge_pred, edge_true)
                        n_edge_terms += 1

                        # Negative edges
                        n_nodes = h.size(0)
                        n_neg = min(ei.size(1), n_nodes * 2)
                        neg_src = torch.randint(0, n_nodes, (n_neg,), device=h.device)
                        neg_dst = torch.randint(0, n_nodes, (n_neg,), device=h.device)
                        neg_ei = torch.stack([neg_src, neg_dst])
                        neg_pred = gae_model.decode_edges(h, neg_ei, rel)
                        if neg_pred is not None:
                            neg_true = torch.zeros_like(neg_pred)
                            l_edge = l_edge + F.binary_cross_entropy(neg_pred, neg_true)
                            n_edge_terms += 1

        if n_edge_terms > 0:
            l_edge = l_edge / n_edge_terms
        loss_dict["edge_recon"] = l_edge.item()

        total = l_node + self.lambda_graph * l_graph + self.lambda_edge * l_edge
        loss_dict["total"] = total.item()

        return total, loss_dict
