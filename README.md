<div align="center">

# Hopf-GraphVAE
## Physics-Informed Graph Neural Network for Normative Brain Dynamics

### Anomaly Detection in Major Depressive Disorder via Stuart-Landau–Grounded Graph Variational Autoencoder with Two-Level Fisher LDA Scoring

[![Python](https://img.shields.io/badge/Python-≥3.9-3776AB?logo=python&logoColor=white)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-≥2.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/) [![PyG](https://img.shields.io/badge/PyTorch_Geometric-≥2.4-3C2179?logo=pyg&logoColor=white)](https://pyg.org/) [![License](https://img.shields.io/badge/License-Academic_Use-lightgrey)]() [![Status](https://img.shields.io/badge/Status-v4_Final-brightgreen)]() [![Atlas](https://img.shields.io/badge/Parcellation-216_ROI-blue)]() [![Architecture](https://img.shields.io/badge/Params-8%2C515_total-orange)]()

</div>

---

## Overview

This repository contains the **Hopf-GraphVAE**, a physics-informed deep learning architecture that detects depression-related dynamical abnormalities without ever training on depressed brains. Rather than framing MDD detection as binary classification (which fails at $n = 19$), the model learns a **normative manifold** of healthy brain dynamics and scores MDD subjects by how far they deviate from it.

The key innovations:

1. **Biophysically grounded node features** — every node carries the per-region bifurcation parameter $a_j$, natural frequency $\omega_j$, and goodness-of-fit $\chi^2_j$ estimated by the Stuart-Landau / Hopf bifurcation framework via the [UKF-MDD](https://github.com/skaraoglu/UKF-MDD) pipeline, plus Yeo 7-network one-hot encodings (11 features per ROI).

2. **Multi-relational graph attention** — three edge types (PLV phase synchrony, MVAR Granger causality, SC structural connectivity) with learned per-relation attention weights.

3. **MLP edge decoders on frozen encoder embeddings** — per-relation edge prediction using $|\mathbf{h}_i - \mathbf{h}_j|$ as input, bypassing the VAE bottleneck entirely.

4. **Two-level Fisher Linear Discriminant scoring** — data-driven combination of node dynamics and edge connectivity anomalies, with per-relation weighting at level 1 and node-vs-edge weighting at level 2. No manual tuning.

---

## The $n = 19$ Problem

<table>
<tr>
<td width="50%">

**Classification (insufficient data)**
- Binary classification: active NF vs. sham
- Trained on 18 subjects per fold
- Memorized subject identity → Cohen's $d$ of $-6$ to $-11$
- Implausibly large vs. UKF reference $d = -0.835$
- **Verdict:** classification fails at this sample size

</td>
<td width="50%">

**Normative anomaly detection (this work)**
- Train exclusively on healthy controls ($n = 295$ sessions)
- MDD subjects are test-only — never seen during training
- Overfitting eliminated by construction
- HC vs MDD: $d = +3.23$, $p = 1.7 \times 10^{-10}$
- All 4 intervention scales survive FDR correction
- **Verdict:** "how far from healthy?" not "which group?"

</td>
</tr>
</table>

---

## Architecture

<div align="center">
<img src="docs/architecture_dark.svg" alt="Hopf-GraphVAE Architecture" width="900"/>
</div>

### Node Features (11-dimensional per ROI)

| Feature | Dim | Source | Meaning |
|---------|-----|--------|---------|
| $a_j$ | 1 | [UKF-MDD](https://github.com/skaraoglu/UKF-MDD) | Bifurcation parameter — distance from critical point |
| $\omega_j$ | 1 | Hilbert phase | Natural oscillation frequency (Hz) |
| $\chi^2_j$ | 1 | UKF fit | Goodness-of-fit (model–data agreement) |
| Network one-hot | 8 | Yeo 7 + Subcortical | Functional network membership |

### Edge Types (3 relations)

| Relation | Type | Source | Encoder Weight (Conv₁) | Edge Decoder $d$ | Fisher Weight |
|----------|------|--------|------------------------|-------------------|---------------|
| **PLV** | Undirected | Phase Locking Value | 0.771 | +0.38 | +0.17 |
| **SC** | Undirected | $\exp(-d/40\text{mm})$ | 0.185 | +0.68 | +0.31 |
| **MVAR** | Directed | Lasso-MVAR | 0.044 | +1.17 | +0.53 |

The encoder and edge decoders use these three relations differently: the encoder (Conv₁) relies primarily on PLV for message passing, while the edge decoders find MVAR most discriminative for HC-vs-MDD separation. Both sets of weights are learned from data.

---

## Model Components

### Multi-Relational Graph Attention Convolution

Each GAT layer maintains separate learnable projections $W_r$ and attention vectors $\mathbf{a}_r$ for each edge relation $r \in \{\text{PLV}, \text{MVAR}, \text{SC}\}$:

$$h_j^{(l+1)} = \text{ELU}\!\left( \frac{1}{|R|} \sum_{r \in R} \sum_{i \in \mathcal{N}_r(j)} \alpha_{ij}^{(r)} \, W_r \, h_i^{(l)} \right)$$

### Frozen Encoder (5,485 parameters)

Two multi-relational GAT layers ($11 \to 32 \to 32$) with a masked residual connection produce per-ROI embeddings $\mathbf{h}_j \in \mathbb{R}^{32}$. The masked residual projects the input through `input_proj` ($11 \to 32$) but **zeros the physics features** $(a_j, \omega_j, \chi^2_j)$ — forcing the encoder to reconstruct dynamics through graph message passing rather than shortcutting via identity. A physics head ($32 \to 16 \to 1$) validates the encoder during pre-training ($R^2 = 0.983$). The encoder is frozen after pre-training.

### Trainable Graph VAE (3,030 parameters)

**Node path:** VAE projections ($\mathbf{h}_j \to \mu_j, \log\sigma^2_j \in \mathbb{R}^8$) → reparameterized $z_j$ → node decoder ($8 \to 32 \to 16 \to 3$) → reconstructed $(\hat{a}_j, \hat{\omega}_j, \hat{\chi}^2_j)$.

**Edge path:** Three MLP edge decoders predict edge existence from $|\mathbf{h}_i - \mathbf{h}_j|$ for PLV, SC, and MVAR independently. Each MLP is $32 \to 16 \to 1$ with ELU activation. Edge decoders use frozen $\mathbf{h}$ (not $z$), making them functional regardless of KL state.

| Component | Shape | Parameters | Status |
|-----------|-------|------------|--------|
| $f_\mu, f_{\sigma}$ (bottleneck) | $32 \to 8$ each | 528 | Trainable |
| Node decoder | $8 \to 32 \to 16 \to 3$ | 867 | Trainable |
| Edge decoders (PLV, SC, MVAR) | $32 \to 16 \to 1$ each | 1,635 | Trainable |
| **Total trainable** | | **3,030** | |

### Two-Level Fisher LDA Scoring

**Level 1 — Per-relation edge weighting:** Each edge type gets a signed Fisher weight $w_r = d_r / \sum|d_r|$ derived from its individual HC-vs-MDD effect size. The composite edge score is $\sum w_r \cdot z_r$.

**Level 2 — Node-vs-edge weighting:** The composite edge score and node reconstruction error are combined with signed Fisher weights: $S = w_\text{node} \cdot z_\text{node} + w_\text{edge} \cdot z_\text{edge}$.

Both levels are fully data-driven. Signed weights handle reversed signals naturally — if a component separates groups in the negative direction, its negative weight flips the contribution automatically.

**Loss function (GVAE training):**

$$\mathcal{L} = \underbrace{2(a - \hat{a})^2 + (\omega - \hat{\omega})^2 + (\chi^2 - \hat{\chi}^2)^2}_{\text{feature-weighted node recon}} + \; 0.5 \cdot \underbrace{\sum_{r} \text{BCE}(\hat{A}_r, A_r)}_{\text{3-relation edge recon}} + \; \beta \cdot \underbrace{D_{\text{KL}}(q \| \mathcal{N}(0, I))}_{\text{cyclical } \beta\text{-annealing}}$$

---

## Parameter Budget

```
Total parameters:                            8,515
├── Frozen encoder:                          5,485  (64%)
│   ├── conv1 (3-relation GAT, 11→32):       1,286
│   ├── conv2 (3-relation GAT, 32→32):       3,302
│   ├── input_proj (masked residual, 11→32):   352
│   └── physics_head (32→16→1):                545
└── Trainable GVAE:                          3,030  (36%)
    ├── fc_μ, fc_σ² (32→8 each):               528
    ├── node_decoder (8→32→16→3):              867
    └── edge_decoders (3 × MLP 32→16→1):     1,635
```

---

## Data Isolation

```
┌────────────┬─────────────────┬──────────────┬──────────────┬──────────────┐
│  Synthetic │    HC train     │   HC test    │  MDD rest1   │  MDD rest2   │
│  n = 200   │ 24 subj (235s)  │ 6 subj (60s) │   19 subj    │   19 subj    │
│  Stage 1   │    Stage 2      │  Test only   │  Test only   │  Test only   │
└────────────┴─────────────────┴──────────────┴──────────────┴──────────────┘
 Synthetic + HC train = train  |  HC test + MDD = test (never trained on)
```

The HC train/test split is **by subject** (not session) to prevent leakage. MDD subjects are never seen during any training stage. HC train vs. test overfitting check: $p = 0.77$.

---

## Key Design Decisions

**Node-level (not graph-level) bottleneck** — Graph-level pooling caused KL collapse ($D_{\text{KL}} \to 0.0002$) because a single $z$ vector for the whole graph could be bypassed by the frozen encoder embeddings. Node-level bottleneck gives each ROI its own $z_j$, forcing per-ROI dynamical information through the bottleneck.

**Edge decoders on $\mathbf{h}$ (not $z$)** — The bilinear form $\sigma(z_i^\top W z_j)$ has vanishing gradients at $z \approx 0$ (a saddle point), making it non-functional when KL collapses. MLP decoders on $|\mathbf{h}_i - \mathbf{h}_j|$ use the rich 32-dim frozen encoder output directly, decoupled from the VAE bottleneck.

**Absolute difference $|\mathbf{h}_i - \mathbf{h}_j|$ (not concatenation)** — Concatenation $[\mathbf{h}_i \| \mathbf{h}_j]$ gives the decoder access to individual node magnitudes, allowing it to exploit the fact that MDD $\mathbf{h}$ vectors have different norms — a confound rather than a connectivity signal. Absolute difference isolates the pairwise relationship.

**Per-relation Fisher weighting** — PLV, SC, and MVAR carry different amounts of discriminative information (edge $d$ ranges from +0.38 to +1.17). Equal weighting dilutes the strongest signals. Fisher LDA weights each relation proportional to its HC-vs-MDD effect size, maximizing the combined test statistic.

**Feature-weighted reconstruction** — Weights $[2, 1, 1]$ on $(a, \omega, \chi^2)$ emphasize the bifurcation parameter, the UKF pipeline's primary clinical marker.

**Cyclical $\beta$-annealing** — Four cycles of $\beta$ from $0 \to 0.5$ following Fu et al. (2019), preventing KL collapse during early training while allowing the encoder to learn useful representations.

---

## Key Results

| Metric | Value | 95% CI | UKF Reference |
|--------|-------|--------|---------------|
| HC vs MDD separation | $d = +3.23$, $p = 1.7 \times 10^{-10}$ | — | — |
| Overfitting check | $p = 0.77$ | — | — |
| Whole-brain intervention | $d = +1.43$, FDR $p = 0.021^*$ | $[0.60, 2.86]$ | $d = -0.84$, $p = 0.080$ |
| Circuit intervention | $d = +1.34$, FDR $p = 0.021^*$ | $[0.51, 2.65]$ | $d = -1.09$, $p = 0.027$ |
| Limbic intervention | $d = +1.34$, FDR $p = 0.021^*$ | $[0.58, 2.48]$ | — |
| Subcortical intervention | $d = +1.60$, FDR $p = 0.021^*$ | $[0.58, 4.36]$ | — |
| Circuit enrichment (top-10) | 2.19× (7/10 circuit ROIs) | — | — |
| Circuit enrichment (top-15) | 1.88× (9/15 circuit ROIs) | — | — |
| Heterogeneity (raw $a$, circuit) | $d = +1.44$, $p = 0.008$ | — | $d = +1.01$, $p = 0.042$ |
| #1 anomalous ROI | RH Default PFCdPFCm₄ | — | Converges with Ch. 5 cluster |
| #1 anomalous network | Limbic | — | — |

All four intervention scales survive Benjamini-Hochberg FDR correction. Bootstrap 95% CIs computed from 10,000 resamples. Active group moves **away** from HC (increased anomaly), sham moves **toward** HC (decreased anomaly).

### Top 15 Anomalous ROIs

| Rank | ROI | Network | Circuit? |
|------|-----|---------|----------|
| 1 | RH Default PFCdPFCm₄ | Default Mode | ✓ |
| 2 | LH Limbic TempPole₁ | Limbic | ✓ |
| 3 | LH Limbic TempPole₂ | Limbic | ✓ |
| 4 | LH Cont Cing₂ | Frontoparietal | |
| 5 | RH SalVentAttn FrOperIns₁ | Salience/VentAttn | |
| 6 | RH Limbic TempPole₁ | Limbic | ✓ |
| 7 | LH Default Temp₅ | Default Mode | ✓ |
| 8 | Thal-rh | Subcortical | ✓ |
| 9 | NAcc-rh | Subcortical | ✓ |
| 10 | LH Default Par₁ | Default Mode | |
| 11 | LH Limbic TempPole₄ | Limbic | ✓ |
| 12 | LH SalVentAttn FrOperIns₂ | Salience/VentAttn | |
| 13 | RH SalVentAttn FrOperIns₂ | Salience/VentAttn | |
| 14 | RH Limbic TempPole₂ | Limbic | ✓ |
| 15 | RH Cont Cing₂ | Frontoparietal | |

---

## Upstream Dependencies

The Hopf-GraphVAE consumes outputs from the R biophysical pipeline ([UKF-MDD](https://github.com/skaraoglu/UKF-MDD)):

| Input | File | Format |
|-------|------|--------|
| Bifurcation parameters | `results/v3/sl_stage1_results_216roi.csv` | CSV (one row per ROI per subject per session) |
| PLV matrices | `results/v3/plv/plv_all_216roi.rds` | R list, keyed `"subject_id\|session"` |
| MVAR matrices | `results/v3/s2_mvar_all_216roi.rds` | R list, keyed `"subject_id\|session"` |
| HC comparison data | `results/ch5_v4def/ch5_v4def_results.rds` | R list |

---

## Requirements

<table>
<tr>
<td>

**Python packages**
```
torch, torch_geometric,
numpy, pandas, scipy,
statsmodels, pyreadr,
scikit-learn, matplotlib
```

</td>
<td>

**Upstream (R pipeline)**
```
pracma, MASS, Matrix,
dplyr, tidyr, ggplot2,
scales, glmnet, igraph,
parallel, zoo
```

</td>
</tr>
</table>

**System:** Python ≥ 3.9 · PyTorch ≥ 2.0 · PyTorch Geometric ≥ 2.4 · R ≥ 4.2 (for upstream pipeline only)

---

## Quick Start

```bash
# 1. Ensure upstream pipeline has been run
# (github.com/skaraoglu/UKF-MDD)

# 2. Install Python dependencies
pip install torch torch_geometric pyreadr scikit-learn statsmodels

# 3. Run the full pipeline
jupyter execute main_analysis.ipynb

# Pipeline stages:
#   S1–S6:   Data loading, graph construction, quality control
#   S7–S10:  Synthetic pre-training (encoder, 100 epochs)
#   S11–S12: HC data loading, GVAE training (200 epochs)
#   S13:     Anomaly scoring (two-level Fisher LDA)
#   S14:     Statistical analysis (FDR, bootstrap CIs, heterogeneity)
```

---

## Citation

If you use this architecture or build on this work, please cite:

---

<div align="center">

*Built with [PyTorch Geometric](https://pyg.org/) · Node dynamics from [UKF-MDD](https://github.com/skaraoglu/UKF-MDD) · Parcellation: [Schaefer 2018](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal) + [Melbourne Subcortex](https://github.com/yetianmed/subcortex)*

</div>
