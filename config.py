# =============================================================================
# config.py — All Constants and Paths
#
# v4: Incorporates all 8 improvements.
#     - Classification constants REMOVED (no OutputHeads, no NetworkHierarchicalPool)
#     - Feature reconstruction weights ADDED (Improvement 4)
#     - Free-bits KL threshold ADDED (Improvement 7)
#     - HC MVAR path ADDED
#     - Multi-scale circuit ROI indices referenced from roi_meta at runtime
# =============================================================================

import logging
from pathlib import Path

import numpy as np
import torch

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hopf_stgnn")

# ─── Reproducibility ───────────────────────────────────────────────────────
SEED = 42

# ─── Project Paths ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(".")

# Input data (R pipeline outputs)
MDD_UKF_CSV    = PROJECT_ROOT / "results" / "v3" / "sl_stage1_results_216roi.csv"
PLV_RDS        = PROJECT_ROOT / "results" / "v3" / "plv" / "plv_all_216roi.rds"
MVAR_RDS       = PROJECT_ROOT / "results" / "v3" / "s2_mvar_all_216roi.rds"
GROUP_CSV      = PROJECT_ROOT / "data" / "parcellated" / "group_assignments.csv"
TOPO_CSV       = PROJECT_ROOT / "results" / "v3" / "s3_topology_216roi.csv"

# Chapter 5 results
CH5_RESULTS    = PROJECT_ROOT / "results" / "ch5_v4def" / "ch5_v4def_results.rds"
CH5_SUPP_V2    = PROJECT_ROOT / "results" / "ch5_v4def" / "ch5_supplement_v2_results.rds"

# HC MVAR results (produced by hc_mvar.ipynb)
HC_MVAR_RDS    = PROJECT_ROOT / "results" / "ch5_v4def" / "hc_mvar.rds"

# Atlas files
SCHAEFER_NII   = PROJECT_ROOT / "atlases" / "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"
MELB_NII       = PROJECT_ROOT / "atlases" / "Tian_Subcortex_S1_3T_FSLMNI152_2mm.nii.gz"

# HC BOLD data
HC_216_DIR     = PROJECT_ROOT / "data" / "parcellated_hc" / "216roi"

# Output directories
RESULTS_DIR    = PROJECT_ROOT / "results" / "hopf_stgnn"
RESULTS_V4     = RESULTS_DIR / "v4"

# ─── Acquisition Constants ─────────────────────────────────────────────────
TR             = 2.0
N_VOLS         = 260
BOLD_FREQ_MIN  = 0.01
BOLD_FREQ_MAX  = 0.10
N_ROIS_216     = 216
N_ROIS_110     = 110         # Harvard-Oxford atlas

# ─── Stuart-Landau Constants (from R/sl_models.R) ───────────────────────────
A_MIN          = -2.0        # SL_BOUNDS$A_MIN
A_MAX          =  2.0        # SL_BOUNDS$A_MAX
OM_MIN         = 2 * np.pi * BOLD_FREQ_MIN * TR   # ≈ 0.1257
OM_MAX         = 2 * np.pi * BOLD_FREQ_MAX * TR   # ≈ 1.2566

# ─── SC Matrix Constants ────────────────────────────────────────────────────
SC_LAMBDA_MM   = 40.0        # Exponential decay length scale (mm)

# ─── Yeo 7-Network Labels ─────────────────────────────────────────────────
YEO_NETWORKS = {
    "Vis":         "Visual",
    "SomMot":      "Somatomotor",
    "DorsAttn":    "Dorsal Attention",
    "SalVentAttn": "Salience/VentAttn",
    "Limbic":      "Limbic",
    "Cont":        "Frontoparietal",
    "Default":     "Default Mode",
}
SUBCORTICAL_LABEL = "Subcortical"

# Depression-circuit ROI patterns
MDD_CIRCUIT_SUBCORT = [
    "Amyg-lh", "Amyg-rh", "Hipp-lh", "Hipp-rh",
    "NAcc-lh", "NAcc-rh", "Thal-lh", "Thal-rh",
]
MDD_CIRCUIT_CORTICAL_PATTERNS = [
    "Default_PFC", "Cont_PFCl", "SalVentAttn_Med",
    "Default_Temp", "Limbic_OFC", "Limbic_TempPole",
]

# ─── Graph Construction ───────────────────────────────────────────────────
EDGE_RELATIONS = ["plv", "mvar", "sc"]
RELATION_KEYS  = [f"edge_index_{r}" for r in EDGE_RELATIONS]
PLV_TOP_K      = 10.0   # top-k% density for PLV thresholding
SC_TOP_K       = 15.0   # top-k% density for SC thresholding

# ─── Encoder Architecture ─────────────────────────────────────────────────
HIDDEN_DIM     = 32
DROPOUT        = 0.1
N_NODE_FEATURES_OUT = 3   # (a, omega, chisq) for GVAE decoder

# ─── Synthetic Pre-Training ───────────────────────────────────────────────
N_SYN          = 200
N_VAL_SYN      = 20
PRETRAIN_EPOCHS = 100
PRETRAIN_LR    = 3e-3
PRETRAIN_WD    = 1e-4
PRETRAIN_BS    = 8
LAMBDA_PHYSICS = 1.0
LAMBDA_SUBCRIT = 0.05

# ─── GVAE Training ────────────────────────────────────────────────────────
LATENT_DIM     = 8       # Tighter bottleneck (was 16 in v3)
GAE_EPOCHS     = 200     # Extended from 100 for edge decoder convergence
GAE_LR         = 1e-3
GAE_WD         = 1e-4
GAE_BS         = 8
LAMBDA_EDGE    = 0.5     # Edge decoder training weight (h-based MLP, independent of z)
ALPHA_EDGE     = 0.3     # Edge contribution to anomaly score (node=0.7, edge=0.3)
                         # Proportional to signal: node d=2.75, edge d=0.57

# Improvement 4: Feature-specific reconstruction weights [a, omega, chisq]
RECON_FEATURE_WEIGHTS = torch.tensor([2.0, 1.0, 1.0])

# Improvement 7: Free-bits KL schedule (replaces linear beta annealing)
BETA_MAX       = 0.5     # Standard — edge decoders decoupled from z, no need to reduce
BETA_WARMUP    = 30      # Used within each cycle (ratio parameter)
BETA_N_CYCLES  = 4       # Number of cyclical annealing cycles (Fu et al. 2019)
BETA_CYCLE_RATIO = 0.5   # Fraction of each cycle spent ramping (rest holds at BETA_MAX)
FREE_BITS      = 0.0     # Disabled — cyclical beta-annealing used instead

# ─── Anomaly Detection ────────────────────────────────────────────────────
OUTLIER_N_SD   = 20       # Outlier threshold: HC_mean + OUTLIER_N_SD * HC_std

# ─── Statistical Analysis ─────────────────────────────────────────────────
N_PERMS        = 10000
