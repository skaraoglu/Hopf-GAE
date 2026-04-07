# =============================================================================
# models.py — All Model Architectures (v4b)
#
# v4b improvements applied:
#   0. Classification branch REMOVED (OutputHeads, NetworkHierarchicalPool)
#   1. Learned relation-importance weights in MultiRelationalGATConv
#   2. Edge-attribute-aware attention in MultiRelationalGATConv
#   3. Residual input projection in HopfEncoder
#   4. Feature-weighted reconstruction loss in GAELoss
#   5. Decoder skip REMOVED — z-only decoder (v3b proven, prevents KL collapse)
#   6. Per-relation bilinear edge decoders in HopfGraphVAE
#   7. Free-bits REMOVED — cyclical beta annealing used instead
#
# Classes:
#   MultiRelationalGATConv — GAT with per-relation weights + edge attributes
#   PhysicsAuxHead         — Per-node bifurcation parameter predictor
#   HopfPhysicsLoss        — Physics-only pre-training loss (no BCE)
#   HopfEncoder            — Two-layer GAT encoder with residual connection
#   HopfGraphVAE           — Node-level GVAE with z-only decoder + bilinear edges
#   GAELoss                — Feature-weighted recon + standard KL + multi-edge
# =============================================================================

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

from config import RELATION_KEYS, RECON_FEATURE_WEIGHTS, FREE_BITS


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
# 4 — GRAPH VARIATIONAL AUTOENCODER
#
# Improvements 5 + 6:
#   5. Decoder skip connection from frozen encoder h
#   6. Per-relation bilinear edge decoders (PLV + SC)
# =============================================================================

class HopfGraphVAE(nn.Module):
    """
    Graph VAE with node-level bottleneck and z-only decoder.

    Encoder (frozen): conv1 -> ELU -> conv2 -> ELU -> + residual -> h_j
    Bottleneck:       h_j -> mu_j, logvar_j -> z_j ~ N(mu_j, sigma_j^2)
    Node decoder:     z_j -> MLP -> (a_j, omega_j, chisq_j)
    Edge decoders:    MLP(|h_i - h_j|) -> P(edge_ij) per relation
                      Uses frozen h directly — functional regardless of z/KL state
    """

    def __init__(self, encoder_model: HopfEncoder,
                 latent_dim: int = 8, n_node_features_out: int = 3):
        super().__init__()

        # Freeze the entire encoder
        self.encoder = encoder_model
        for param in self.encoder.parameters():
            param.requires_grad = False

        hidden_dim = encoder_model.hidden_dim

        # Node-level VAE bottleneck
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Node decoder: z-only input (no skip from h_frozen)
        self.node_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ELU(),
            nn.Linear(hidden_dim // 2, n_node_features_out),
        )

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

    def encode(self, data):
        """Run frozen encoder, return h, mu, logvar per node."""
        if not hasattr(data, "batch") or data.batch is None:
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)

        with torch.no_grad():
            enc_out = self.encoder(data)
        h = enc_out["node_embeddings"]

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return h, mu, logvar, data.batch

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def decode_nodes(self, z):
        """Decode from z only — no skip connection."""
        return self.node_decoder(z)

    def decode_edges(self, h, edge_index, relation):
        """MLP edge decoder: P(edge_ij) from |h_i - h_j|."""
        if relation not in self.edge_decoders:
            return None
        src, dst = edge_index
        edge_feat = torch.abs(h[src] - h[dst])
        return torch.sigmoid(self.edge_decoders[relation](edge_feat).squeeze(-1))

    def forward(self, data):
        h, mu, logvar, batch = self.encode(data)
        z = self.reparameterize(mu, logvar)
        node_recon = self.decode_nodes(z)  # z-only, no skip

        return {
            "node_recon": node_recon,
            "mu": mu,
            "logvar": logvar,
            "h": h,
            "z": z,
            "batch": batch,
        }


# =============================================================================
# 5 — GAE LOSS
#
# Improvements 4 + 6:
#   4. Feature-weighted node reconstruction loss
#   6. Per-relation edge reconstruction loss
#   Free-bits removed — cyclical beta annealing handles KL scheduling
# =============================================================================

class GAELoss(nn.Module):
    """
    Graph VAE loss with feature-weighted recon, standard KL, and multi-edge.

    L = L_node_weighted + lambda_edge * L_edge_multi + beta * L_KL
    """

    def __init__(self, lambda_edge: float = 0.1, beta: float = 0.01,
                 feature_weights: torch.Tensor = None,
                 free_bits: float = 0.0):
        super().__init__()
        self.lambda_edge = lambda_edge
        self.beta = beta
        self.free_bits = free_bits

        if feature_weights is not None:
            self.register_buffer("feature_weights", feature_weights)
        else:
            self.register_buffer("feature_weights", RECON_FEATURE_WEIGHTS)

    def forward(self, result, data, gvae_model=None):
        loss_dict = {}

        # ── Improvement 4: Feature-weighted node reconstruction ──────────
        target_nodes = data.x[:, :3]
        diff_sq = (result["node_recon"] - target_nodes) ** 2  # (N, 3)
        w = self.feature_weights.to(diff_sq.device)
        l_node = (diff_sq * w).mean()
        loss_dict["node_recon"] = l_node.item()

        # ── Improvement 7: Free-bits KL divergence ───────────────────────
        mu = result["mu"]
        logvar = result["logvar"]
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (N, latent_dim)

        if self.free_bits > 0:
            # Only penalise KL above the free-bits threshold per dimension
            kl_clamped = torch.clamp(kl_per_dim - self.free_bits, min=0.0)
            kl = kl_clamped.mean()
        else:
            kl = kl_per_dim.mean()
        loss_dict["kl"] = kl.item()
        # Also log the raw (unclamped) KL for monitoring
        loss_dict["kl_raw"] = kl_per_dim.mean().item()

        # ── Per-relation edge reconstruction (using h, not z) ──────────────
        h = result["h"]
        batch = result["batch"]
        l_edge = torch.tensor(0.0, device=h.device)
        n_edge_terms = 0

        if gvae_model is not None:
            for rel in ["plv", "sc", "mvar"]:
                ei_key = f"edge_index_{rel}"
                if hasattr(data, ei_key) and getattr(data, ei_key).numel() > 0:
                    ei = getattr(data, ei_key)
                    edge_pred = gvae_model.decode_edges(h, ei, rel)
                    if edge_pred is not None:
                        # Target: all observed edges are positive (1.0)
                        edge_true = torch.ones_like(edge_pred)
                        l_edge = l_edge + F.binary_cross_entropy(edge_pred, edge_true)
                        n_edge_terms += 1

                        # Sample negative edges for balanced loss
                        n_nodes = h.size(0)
                        n_neg = min(ei.size(1), n_nodes * 2)
                        neg_src = torch.randint(0, n_nodes, (n_neg,), device=h.device)
                        neg_dst = torch.randint(0, n_nodes, (n_neg,), device=h.device)
                        neg_ei = torch.stack([neg_src, neg_dst])
                        neg_pred = gvae_model.decode_edges(h, neg_ei, rel)
                        if neg_pred is not None:
                            neg_true = torch.zeros_like(neg_pred)
                            l_edge = l_edge + F.binary_cross_entropy(neg_pred, neg_true)
                            n_edge_terms += 1

        if n_edge_terms > 0:
            l_edge = l_edge / n_edge_terms
        loss_dict["edge_recon"] = l_edge.item()

        total = l_node + self.lambda_edge * l_edge + self.beta * kl
        loss_dict["total"] = total.item()

        return total, loss_dict
