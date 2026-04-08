# =============================================================================
# utils.py — All Utility Functions (v4)
#
# Data loading, RDS reading, ROI metadata, graph construction,
# HC graph building (with MVAR support), simulator, batching helpers.
# =============================================================================

import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.signal import hilbert
from scipy.spatial.distance import pdist, squareform
from torch_geometric.data import Data

from config import (
    SEED, TR, N_VOLS, N_ROIS_216, BOLD_FREQ_MIN, BOLD_FREQ_MAX,
    A_MIN, SC_LAMBDA_MM, PLV_TOP_K, SC_TOP_K,
    YEO_NETWORKS, SUBCORTICAL_LABEL,
    MDD_CIRCUIT_SUBCORT, MDD_CIRCUIT_CORTICAL_PATTERNS,
    EDGE_RELATIONS, PROJECT_ROOT,
    MDD_UKF_CSV, PLV_RDS, MVAR_RDS, GROUP_CSV, TOPO_CSV,
    CH5_RESULTS, CH5_SUPP_V2, HC_MVAR_RDS,
    SCHAEFER_NII, MELB_NII, HC_216_DIR,
    log,
)


# =============================================================================
# 1 — REPRODUCIBILITY
# =============================================================================

def seed_everything(seed: int = SEED):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# 2 — RDS READER
# =============================================================================

_rds_reader = None

try:
    import rdata as _rdata_mod
    _rds_reader = "rdata"
except ImportError:
    pass

if _rds_reader is None:
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri, numpy2ri
        pandas2ri.activate()
        numpy2ri.activate()
        _rds_reader = "rpy2"
    except Exception:
        pass

if _rds_reader is None:
    try:
        import pyreadr
        _rds_reader = "pyreadr"
    except ImportError:
        pass


def _sparse_obj_to_dense(obj: Any) -> Optional[np.ndarray]:
    """Convert R sparse matrix to dense numpy array."""
    from scipy.sparse import csc_matrix, coo_matrix

    def _try_convert(d):
        def _get(key):
            return d.get(key) if isinstance(d, dict) else getattr(d, key, None)

        dim = _get("Dim") or _get("dim")
        if dim is None:
            return None
        try:
            dim_arr = np.asarray(dim)
            nrow, ncol = int(dim_arr.flat[0]), int(dim_arr.flat[1])
        except (TypeError, IndexError, ValueError):
            return None

        x = _get("x")
        x_arr = np.asarray(x, dtype=np.float64) if x is not None else None
        i_slot, p_slot, j_slot = _get("i"), _get("p"), _get("j")

        if i_slot is not None and p_slot is not None:
            i_arr = np.asarray(i_slot, dtype=np.int32)
            p_arr = np.asarray(p_slot, dtype=np.int32)
            if x_arr is None:
                x_arr = np.ones(len(i_arr), dtype=np.float64)
            return csc_matrix((x_arr, i_arr, p_arr), shape=(nrow, ncol)).toarray()

        if i_slot is not None and j_slot is not None:
            i_arr = np.asarray(i_slot, dtype=np.int32)
            j_arr = np.asarray(j_slot, dtype=np.int32)
            if x_arr is None:
                x_arr = np.ones(len(i_arr), dtype=np.float64)
            return coo_matrix((x_arr, (i_arr, j_arr)), shape=(nrow, ncol)).toarray()
        return None

    result = _try_convert(obj)
    if result is not None:
        return result
    if hasattr(obj, "__dict__"):
        result = _try_convert(vars(obj))
        if result is not None:
            return result
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, dict) or hasattr(v, "Dim"):
                result = _try_convert(v)
                if result is not None:
                    return result
    return None


def _rdata_coerce_value(v: Any) -> Any:
    """Coerce rdata value to numpy array if possible."""
    if hasattr(v, "__array__"):
        return np.asarray(v)
    dense = _sparse_obj_to_dense(v)
    if dense is not None:
        return dense
    if isinstance(v, dict) and "A" in v:
        mat = v["A"]
        if hasattr(mat, "values"):
            return np.asarray(mat.values)
        elif hasattr(mat, "__array__"):
            return np.asarray(mat)
    if isinstance(v, (list, tuple)):
        try:
            return np.asarray(v, dtype=np.float64)
        except (ValueError, TypeError):
            return v
    return v


def read_rds(path) -> Any:
    """Read an RDS file using the best available reader."""
    path = Path(path)
    if not path.exists():
        log.warning("RDS not found: %s", path)
        return None
    if _rds_reader is None:
        log.warning("No RDS reader available")
        return None
    try:
        if _rds_reader == "rdata":
            import rdata as _rd
            result = _rd.read_rds(str(path))
            dense = _sparse_obj_to_dense(result)
            if dense is not None:
                return dense
            if isinstance(result, dict):
                return {k: _rdata_coerce_value(v) for k, v in result.items()}
            return _rdata_coerce_value(result)
        elif _rds_reader == "rpy2":
            obj = ro.r["readRDS"](str(path))
            if hasattr(obj, "names") and obj.names is not None:
                out = {}
                for name in obj.names:
                    element = obj.rx2(name)
                    out[name] = np.array(element) if hasattr(element, "__array__") else element
                return out
            return np.array(obj) if hasattr(obj, "__array__") else obj
        elif _rds_reader == "pyreadr":
            result = pyreadr.read_r(str(path))
            if len(result) == 0:
                return None
            if len(result) == 1:
                return list(result.values())[0]
            return dict(result)
    except Exception as e:
        log.error("Failed to read RDS %s: %s", path, e)
        return None


# =============================================================================
# 3 — DATA LOADING
# =============================================================================

def load_all_data() -> Dict[str, Any]:
    """Load all R pipeline outputs: UKF, PLV, MVAR, group assignments, topology."""
    result = {}

    # UKF
    if MDD_UKF_CSV.exists():
        result["ukf_df"] = pd.read_csv(MDD_UKF_CSV)
        log.info("UKF: %d rows, %d subjects", len(result["ukf_df"]),
                 result["ukf_df"]["subject"].nunique())
    else:
        result["ukf_df"] = None
        log.error("UKF not found: %s", MDD_UKF_CSV)

    # Groups
    result["group_df"] = pd.read_csv(GROUP_CSV) if GROUP_CSV.exists() else None

    # PLV
    result["plv_all"] = read_rds(PLV_RDS) if PLV_RDS.exists() else None
    if result["plv_all"] and isinstance(result["plv_all"], dict):
        log.info("PLV: %d matrices", len(result["plv_all"]))

    # MVAR
    result["mvar_all"] = read_rds(MVAR_RDS) if MVAR_RDS.exists() else None
    if result["mvar_all"] and isinstance(result["mvar_all"], dict):
        log.info("MVAR: %d matrices", len(result["mvar_all"]))

    # Topology
    result["topo_df"] = pd.read_csv(TOPO_CSV) if TOPO_CSV.exists() else None

    return result


# =============================================================================
# 4 — ROI METADATA
# =============================================================================

def parse_network(roi_name: str) -> str:
    """Assign Yeo 7-network label to an ROI."""
    for prefix, label in YEO_NETWORKS.items():
        if re.match(rf"^7Networks_[LR]H_{prefix}", roi_name):
            return label
    return SUBCORTICAL_LABEL


def is_depression_circuit(roi_name: str) -> bool:
    """Check if ROI belongs to the depression circuit."""
    if roi_name in MDD_CIRCUIT_SUBCORT:
        return True
    return any(p in roi_name for p in MDD_CIRCUIT_CORTICAL_PATTERNS)


def build_roi_meta_and_assignment(ukf_df: pd.DataFrame):
    """Build ROI metadata table and network assignment tensor.
    
    Returns: (roi_meta DataFrame, network_assignment tensor, n_networks int)
    """
    roi_names = ukf_df["roi"].unique().tolist()
    network_labels = sorted(set(list(YEO_NETWORKS.values()) + [SUBCORTICAL_LABEL]))
    network_to_id = {n: i for i, n in enumerate(network_labels)}

    records = []
    for idx, roi_name in enumerate(roi_names):
        network = parse_network(roi_name)
        records.append({
            "roi_index": idx,
            "roi_name": roi_name,
            "network": network,
            "network_id": network_to_id[network],
            "atlas_source": "Melbourne" if not roi_name.startswith("7Networks") else "Schaefer",
            "is_depression_circuit": is_depression_circuit(roi_name),
        })

    roi_meta = pd.DataFrame(records)
    network_assignment = torch.tensor(roi_meta["network_id"].values, dtype=torch.long)
    n_networks = roi_meta["network_id"].nunique()

    log.info("ROI metadata: %d ROIs, %d networks, %d circuit ROIs",
             len(roi_meta), n_networks, roi_meta["is_depression_circuit"].sum())
    return roi_meta, network_assignment, n_networks


# =============================================================================
# 5 — GRAPH CONSTRUCTION
# =============================================================================

def matrix_to_edge_index(mat: np.ndarray, threshold: float = 0.0,
                         directed: bool = False,
                         top_k_pct: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert dense adjacency matrix to COO edge_index + edge_attr."""
    mat = np.array(mat, dtype=np.float32)
    np.fill_diagonal(mat, 0.0)

    if top_k_pct is not None:
        vals = np.abs(mat[mat != 0])
        if len(vals) > 0:
            cutoff = np.percentile(vals, 100 - top_k_pct)
            mat[np.abs(mat) < cutoff] = 0.0
    elif threshold > 0:
        mat[np.abs(mat) < threshold] = 0.0

    if not directed:
        mask = np.triu(np.ones_like(mat, dtype=bool), k=1)
        mat_upper = mat * mask
        rows, cols = np.nonzero(mat_upper)
        vals = mat_upper[rows, cols]
        rows_all = np.concatenate([rows, cols])
        cols_all = np.concatenate([cols, rows])
        vals_all = np.concatenate([vals, vals])
    else:
        rows_all, cols_all = np.nonzero(mat)
        vals_all = mat[rows_all, cols_all]

    edge_index = torch.tensor(np.stack([rows_all, cols_all]), dtype=torch.long)
    edge_attr = torch.tensor(vals_all, dtype=torch.float32).unsqueeze(-1)
    return edge_index, edge_attr


def build_node_features(ukf_df, subject, session, roi_meta):
    """Build node feature matrix [a, omega, chisq, network_onehot] for one subject-session."""
    mask = (ukf_df["subject"] == subject) & (ukf_df["session"] == session)
    sub_df = ukf_df[mask].copy()
    if len(sub_df) == 0:
        return None

    sub_df = sub_df.set_index("roi")
    n_rois = len(roi_meta)
    n_networks = roi_meta["network_id"].nunique()
    x = torch.zeros(n_rois, 3 + n_networks, dtype=torch.float32)

    for _, row in roi_meta.iterrows():
        idx = row["roi_index"]
        roi_name = row["roi_name"]
        net_id = row["network_id"]

        if roi_name in sub_df.index:
            r = sub_df.loc[roi_name]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
            x[idx, 0] = float(r["a"])
            x[idx, 1] = float(r["omega"]) if pd.notna(r.get("omega", np.nan)) else 0.0
            x[idx, 2] = float(r["chisq"]) if pd.notna(r.get("chisq", np.nan)) else 0.0
        else:
            x[idx, 0] = ukf_df["a"].mean()
            x[idx, 1] = ukf_df["omega"].mean() if "omega" in ukf_df.columns else 0.0
            x[idx, 2] = ukf_df["chisq"].mean() if "chisq" in ukf_df.columns else 0.0
        x[idx, 3 + net_id] = 1.0
    return x


def build_subject_graph(subject, session, ukf_df, roi_meta,
                        plv_mat=None, mvar_mat=None, sc_mat=None,
                        group=None, plv_top_k=PLV_TOP_K, sc_top_k=SC_TOP_K):
    """Build a complete PyG Data object for one subject-session."""
    x = build_node_features(ukf_df, subject, session, roi_meta)
    if x is None:
        return None

    kw = {"x": x, "num_nodes": x.shape[0],
          "roi_index": torch.arange(x.shape[0], dtype=torch.long),
          "subject": subject, "session": session, "group": group}

    if plv_mat is not None:
        ei, ea = matrix_to_edge_index(plv_mat, directed=False, top_k_pct=plv_top_k)
        kw["edge_index_plv"], kw["edge_attr_plv"] = ei, ea

    if mvar_mat is not None:
        ei, ea = matrix_to_edge_index(mvar_mat, directed=True, threshold=1e-10)
        kw["edge_index_mvar"], kw["edge_attr_mvar"] = ei, ea

    if sc_mat is not None:
        ei, ea = matrix_to_edge_index(sc_mat, directed=False, top_k_pct=sc_top_k)
        kw["edge_index_sc"], kw["edge_attr_sc"] = ei, ea

    if group is not None:
        y = 1 if group.lower().strip() in ("active", "experimental") else 0
        kw["y_group"] = torch.tensor([y], dtype=torch.long)

    return Data(**kw)


# =============================================================================
# 6 — STRUCTURAL CONNECTIVITY
# =============================================================================

def compute_sc_from_centroids(centroids: np.ndarray, lam: float = SC_LAMBDA_MM):
    """Compute SC = exp(-dist/lambda) from ROI centroids, row-normalised."""
    dist_mat = squareform(pdist(centroids, metric="euclidean"))
    SC = np.exp(-dist_mat / lam)
    np.fill_diagonal(SC, 0.0)
    row_sums = SC.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return SC / row_sums


def load_or_build_sc(roi_meta):
    """Load or compute the structural connectivity matrix."""
    n_rois = len(roi_meta)

    # Try ch5_supplement_v2
    if CH5_SUPP_V2.exists():
        try:
            ch5 = read_rds(CH5_SUPP_V2)
            if isinstance(ch5, dict) and "sc_fcm" in ch5:
                sc_info = ch5["sc_fcm"]
                if isinstance(sc_info, dict) and "SC_matrix" in sc_info:
                    SC = np.array(sc_info["SC_matrix"])
                    if SC.shape == (n_rois, n_rois):
                        log.info("SC from ch5_supplement: %s", SC.shape)
                        return SC, None
        except Exception as e:
            log.warning("SC load failed: %s", e)

    # Try NIfTI centroids
    try:
        import nibabel as nib

        def _centroids(path):
            img = nib.load(str(path))
            data, affine = img.get_fdata(), img.affine
            labs = sorted(set(np.unique(data.astype(int))) - {0})
            return np.array([(affine @ np.append(np.argwhere(data == l).mean(0), 1.0))[:3]
                             for l in labs]), labs

        parts = []
        if SCHAEFER_NII.exists():
            c, _ = _centroids(SCHAEFER_NII)
            parts.append(c)
        if MELB_NII.exists():
            c, _ = _centroids(MELB_NII)
            parts.append(c)
        if parts:
            centroids = np.vstack(parts)[:n_rois]
            if len(centroids) < n_rois:
                pad = np.random.randn(n_rois - len(centroids), 3) * 2 + centroids[-1]
                centroids = np.vstack([centroids, pad])
            SC = compute_sc_from_centroids(centroids)
            log.info("SC from atlas centroids: %s", SC.shape)
            return SC, centroids
    except ImportError:
        pass
    except Exception as e:
        log.warning("Centroid computation failed: %s", e)

    # Synthetic fallback
    log.warning("Using SYNTHETIC SC matrix")
    np.random.seed(SEED)
    centroids = np.random.randn(n_rois, 3) * 30
    return compute_sc_from_centroids(centroids), centroids


# =============================================================================
# 7 — STUART-LANDAU SIMULATOR
# =============================================================================

class StuartLandauSimulator:
    """Whole-brain SL simulator for synthetic training data generation."""

    def __init__(self, sc_matrix, n_rois=N_ROIS_216, TR=2.0,
                 n_TRs=N_VOLS, dt=0.1, sigma=0.02):
        self.SC = sc_matrix
        self.n_rois = n_rois
        self.TR = TR
        self.n_TRs = n_TRs
        self.dt = dt
        self.sigma = sigma
        self.steps_per_TR = int(TR / dt)
        self.total_steps = n_TRs * self.steps_per_TR
        self.sc_degree = sc_matrix.sum(axis=1)

    def simulate(self, a_vector, G=0.5, omega_vector=None, seed=None):
        rng = np.random.RandomState(seed)
        N, dt = self.n_rois, self.dt
        omega = omega_vector if omega_vector is not None else rng.uniform(0.03, 0.07, N) * 2 * np.pi
        x = rng.randn(N) * 0.01
        y = rng.randn(N) * 0.01
        bold = np.zeros((self.n_TRs, N), dtype=np.float32)
        sqrt_dt = np.sqrt(dt)

        for t_step in range(1, self.total_steps + 1):
            zsq = x**2 + y**2
            cr = G * (self.SC @ x - x * self.sc_degree)
            ci = G * (self.SC @ y - y * self.sc_degree)
            x = x + (a_vector * x - omega * y - zsq * x) * dt + cr * dt + self.sigma * sqrt_dt * rng.randn(N)
            y = y + (a_vector * y + omega * x - zsq * y) * dt + ci * dt + self.sigma * sqrt_dt * rng.randn(N)
            if t_step % self.steps_per_TR == 0:
                idx = t_step // self.steps_per_TR - 1
                if idx < self.n_TRs:
                    bold[idx, :] = x

        return {"bold": bold, "a_true": a_vector.copy(), "G": G, "omega": omega}

    def compute_plv(self, bold):
        T, N = bold.shape
        phase = np.angle(hilbert(bold, axis=0))
        plv = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            plv[i, :] = np.abs(np.mean(np.exp(1j * (phase[:, i:i+1] - phase)), axis=0))
        np.fill_diagonal(plv, 1.0)
        return plv

    def compute_mvar(self, bold, alpha=0.1):
        from sklearn.linear_model import Lasso
        T, N = bold.shape
        X, Y = bold[:-1], bold[1:]
        B = np.zeros((N, N), dtype=np.float32)
        model = Lasso(alpha=alpha, max_iter=1000, tol=1e-4)
        for j in range(N):
            model.fit(X, Y[:, j])
            B[j, :] = model.coef_
        return B

    def generate_graph(self, a_mean=-0.20, a_std=0.10, G=0.5,
                       seed=None, compute_connectivity=True):
        rng = np.random.RandomState(seed)
        a_vector = np.clip(rng.normal(a_mean, a_std, self.n_rois), A_MIN, 0.0)
        result = self.simulate(a_vector, G=G, seed=seed)
        if compute_connectivity:
            result["plv"] = self.compute_plv(result["bold"])
            result["mvar"] = self.compute_mvar(result["bold"])
        return result


# =============================================================================
# 8 — BATCHING HELPER
# =============================================================================

_NON_TENSOR_KEYS = {"subject", "session", "group"}

def prepare_graph_for_batching(graph, y_classify_value=None):
    """Return a clean clone suitable for PyG DataLoader batching."""
    kw = {"x": graph.x.clone(), "num_nodes": graph.x.size(0)}

    for rel in EDGE_RELATIONS:
        ei_key, ea_key = f"edge_index_{rel}", f"edge_attr_{rel}"
        if hasattr(graph, ei_key) and getattr(graph, ei_key) is not None:
            kw[ei_key] = getattr(graph, ei_key).clone()
            kw[ea_key] = getattr(graph, ea_key).clone()
        else:
            kw[ei_key] = torch.zeros((2, 0), dtype=torch.long)
            kw[ea_key] = torch.zeros((0, 1), dtype=torch.float32)

    for key in graph.keys():
        if key in kw or key in _NON_TENSOR_KEYS or key == "batch":
            continue
        val = graph[key]
        if isinstance(val, torch.Tensor):
            kw[key] = val.clone()

    new_g = Data(**kw)
    if y_classify_value is not None:
        new_g.y_classify = y_classify_value
    elif hasattr(graph, "y_group"):
        new_g.y_classify = graph.y_group.float().squeeze()
    else:
        new_g.y_classify = torch.tensor(0.0)
    return new_g


# =============================================================================
# 9 — HC DATA LOADING (with MVAR support)
# =============================================================================

def compute_plv_from_bold(bold_df, roi_cols):
    """Compute PLV matrix from parcellated BOLD DataFrame."""
    mat = bold_df[roi_cols].values
    valid = ~np.isnan(mat).any(axis=1)
    mat = mat[valid]
    if len(mat) < 30:
        return None
    phase = np.angle(hilbert(mat, axis=0))
    N = mat.shape[1]
    plv = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        plv[i, :] = np.abs(np.mean(np.exp(1j * (phase[:, i:i+1] - phase)), axis=0))
    np.fill_diagonal(plv, 1.0)
    return plv


def estimate_node_features_from_bold(bold_df, roi_cols):
    """Estimate a_proxy and omega from raw BOLD when UKF unavailable."""
    mat = bold_df[roi_cols].values
    valid = ~np.isnan(mat).any(axis=1)
    mat = mat[valid]
    T, N = mat.shape
    a_est = np.zeros(N, dtype=np.float32)
    omega_est = np.zeros(N, dtype=np.float32)
    freqs = np.fft.rfftfreq(T, d=TR)
    bold_mask = (freqs >= BOLD_FREQ_MIN) & (freqs <= BOLD_FREQ_MAX)

    for j in range(N):
        sig = mat[:, j]
        sig = (sig - sig.mean()) / (sig.std() + 1e-12)
        psd = np.abs(np.fft.rfft(sig)) ** 2
        psd_band = psd.copy()
        psd_band[~bold_mask] = 0
        omega_est[j] = 2 * np.pi * freqs[np.argmax(psd_band)] * TR
        acf = np.correlate(sig, sig, mode="full")[T-1:]
        acf = acf / (acf[0] + 1e-12)
        zc = np.where(acf < 0)[0]
        tau = max(zc[0] if len(zc) > 0 else T // 2, 1)
        a_est[j] = -1.0 / tau
    return a_est, omega_est


def load_hc_graphs(roi_meta, network_assignment, n_networks, sc_matrix,
                   include_mvar=False, max_hc=300):
    """Load HC BOLD CSVs and build PyG graphs.
    
    Parameters
    ----------
    include_mvar : bool
        If True, load HC MVAR matrices from HC_MVAR_RDS and attach as edges.
    
    Returns
    -------
    hc_graphs : list of Data
    hc_file_info : list of dict with path/subject/session
    """
    if not HC_216_DIR.exists():
        log.warning("HC directory not found: %s", HC_216_DIR)
        return [], []

    # Load HC MVAR if requested
    hc_mvar_all = None
    if include_mvar and HC_MVAR_RDS.exists():
        hc_mvar_all = read_rds(HC_MVAR_RDS)
        if hc_mvar_all and isinstance(hc_mvar_all, dict):
            log.info("HC MVAR loaded: %d matrices", len(hc_mvar_all))
        else:
            log.info("HC MVAR load returned non-dict or None")
            hc_mvar_all = None

    # Load HC UKF from Ch5 if available
    hc_ukf_data = None
    if CH5_RESULTS.exists():
        try:
            ch5 = read_rds(CH5_RESULTS)
            if isinstance(ch5, dict):
                for k in ["hc_all", "hc_avg", "hc_avg_roi"]:
                    if k in ch5:
                        hc_ukf_data = ch5[k]
                        log.info("HC UKF from ch5 key: %s", k)
                        break
        except Exception:
            pass

    hc_csvs = sorted(HC_216_DIR.glob("*.csv"))
    log.info("HC CSVs: %d", len(hc_csvs))

    hc_file_info = []
    for fp in hc_csvs:
        m = re.match(r"(.+)_session_(\d+)_216roi\.csv", fp.name)
        if m:
            hc_file_info.append({"path": fp, "subject": m.group(1),
                                 "session": int(m.group(2))})

    log.info("HC files parsed: %d (subjects: %d)",
             len(hc_file_info), len(set(f["subject"] for f in hc_file_info)))

    # Determine ROI columns
    sample_df = pd.read_csv(hc_file_info[0]["path"], nrows=2)
    drop = [c for c in sample_df.columns if c.lower() in ("time", "tr", "t", "", "background", "0")]
    known = set(roi_meta["roi_name"].values)
    hc_roi_cols = [c for c in sample_df.columns if c not in drop and c in known]
    log.info("HC ROI columns matched: %d / %d", len(hc_roi_cols), len(roi_meta))

    hc_graphs = []
    subset = hc_file_info[:max_hc]

    for idx, info in enumerate(subset):
        try:
            bold_df = pd.read_csv(info["path"])
            n_rois = len(hc_roi_cols)

            # Node features: try UKF, fallback to BOLD estimation
            a_est, omega_est = None, None
            if hc_ukf_data is not None:
                try:
                    if hasattr(hc_ukf_data, "columns") and "a" in hc_ukf_data.columns:
                        hc_sub = hc_ukf_data[hc_ukf_data["subject"].astype(str) == str(info["subject"])]
                        if "session" in hc_ukf_data.columns:
                            sess_match = hc_sub[hc_sub["session"].astype(str).str.contains(str(info["session"]))]
                            if len(sess_match) > 0:
                                hc_sub = sess_match
                        if len(hc_sub) >= n_rois // 2:
                            hc_idx = hc_sub.set_index("roi")
                            a_est = np.array([hc_idx.loc[r, "a"] if r in hc_idx.index else -0.26
                                              for r in hc_roi_cols], dtype=np.float32)
                            if "omega" in hc_idx.columns:
                                omega_est = np.array([hc_idx.loc[r, "omega"] if r in hc_idx.index else 0.4
                                                      for r in hc_roi_cols], dtype=np.float32)
                except Exception:
                    pass

            if a_est is None:
                a_est, omega_est = estimate_node_features_from_bold(bold_df, hc_roi_cols)
            if omega_est is None:
                _, omega_est = estimate_node_features_from_bold(bold_df, hc_roi_cols)

            # PLV
            plv_mat = compute_plv_from_bold(bold_df, hc_roi_cols)
            if plv_mat is None:
                continue

            # Node features
            x_hc = torch.zeros(n_rois, 3 + n_networks, dtype=torch.float32)
            x_hc[:, 0] = torch.tensor(a_est[:n_rois], dtype=torch.float32)
            x_hc[:, 1] = torch.tensor(omega_est[:n_rois], dtype=torch.float32)
            x_hc[:, 2] = 0.5
            if network_assignment is not None:
                for j in range(min(n_rois, len(network_assignment))):
                    x_hc[j, 3 + network_assignment[j].item()] = 1.0

            # Edges
            ei_plv, ea_plv = matrix_to_edge_index(plv_mat[:n_rois, :n_rois],
                                                   directed=False, top_k_pct=PLV_TOP_K)
            ei_sc, ea_sc = matrix_to_edge_index(sc_matrix[:n_rois, :n_rois],
                                                 directed=False, top_k_pct=SC_TOP_K)

            # MVAR edges (from hc_mvar.rds if available)
            ei_mvar = torch.zeros((2, 0), dtype=torch.long)
            ea_mvar = torch.zeros((0, 1), dtype=torch.float32)
            if hc_mvar_all is not None:
                mvar_key = f"{info['subject']}|session_{info['session']}"
                if mvar_key in hc_mvar_all:
                    mvar_mat = np.asarray(hc_mvar_all[mvar_key])
                    if hasattr(mvar_mat, "shape") and mvar_mat.shape[0] >= n_rois:
                        ei_mvar, ea_mvar = matrix_to_edge_index(
                            mvar_mat[:n_rois, :n_rois], directed=True, threshold=1e-10)

            data = Data(
                x=x_hc,
                edge_index_plv=ei_plv, edge_attr_plv=ea_plv,
                edge_index_mvar=ei_mvar, edge_attr_mvar=ea_mvar,
                edge_index_sc=ei_sc, edge_attr_sc=ea_sc,
                a_true=torch.tensor(a_est[:n_rois], dtype=torch.float32),
                num_nodes=n_rois,
            )
            hc_graphs.append(data)

        except Exception as e:
            if idx < 3:
                log.warning("HC graph %d failed: %s", idx, e)
            continue

        if (idx + 1) % 50 == 0:
            log.info("  Built %d / %d HC graphs", idx + 1, len(subset))

    log.info("HC graphs: %d / %d", len(hc_graphs), len(subset))
    return hc_graphs, hc_file_info[:len(hc_graphs)]


def split_hc_by_subject(hc_graphs, hc_file_info, test_frac=0.2):
    """Split HC graphs 80/20 by subject (not session) for train/test."""
    if len(hc_graphs) == 0:
        return [], []

    subjects = sorted(set(f["subject"] for f in hc_file_info))
    n_test = max(1, int(len(subjects) * test_frac))
    rng = np.random.RandomState(SEED)
    test_subjs = set(rng.choice(subjects, size=n_test, replace=False))

    subj_per_graph = [hc_file_info[i]["subject"] for i in range(len(hc_graphs))]
    train = [g for g, s in zip(hc_graphs, subj_per_graph) if s not in test_subjs]
    test = [g for g, s in zip(hc_graphs, subj_per_graph) if s in test_subjs]

    log.info("HC split: train=%d (%d subj), test=%d (%d subj)",
             len(train), len(subjects) - n_test, len(test), n_test)
    return train, test


# =============================================================================
# 10 — MDD GRAPH ASSEMBLY
# =============================================================================

def build_empirical_graphs(ukf_df, roi_meta, plv_all, mvar_all, sc_matrix):
    """Build PyG graphs for all MDD subject-sessions.
    
    Returns: (empirical_graphs dict, subjects_list, groups_map)
    """
    empirical_graphs = {}
    subjects_list = []
    groups_map = {}

    subjects = sorted(ukf_df["subject"].unique())
    sessions = sorted(ukf_df["session"].unique())

    for subj in subjects:
        subj_group = ukf_df[ukf_df["subject"] == subj]["group"].iloc[0]
        groups_map[subj] = subj_group

        for sess in sessions:
            key_str = f"{subj}|{sess}"
            plv_mat = None
            if plv_all and isinstance(plv_all, dict) and key_str in plv_all:
                plv_mat = np.array(plv_all[key_str])

            mvar_mat = None
            if mvar_all and isinstance(mvar_all, dict) and key_str in mvar_all:
                m = mvar_all[key_str]
                if hasattr(m, "shape"):
                    mvar_mat = np.asarray(m)
                elif isinstance(m, dict) and "A" in m:
                    mat = m["A"]
                    mvar_mat = np.asarray(mat.values if hasattr(mat, "values") else mat)

            graph = build_subject_graph(
                subject=subj, session=sess, ukf_df=ukf_df, roi_meta=roi_meta,
                plv_mat=plv_mat, mvar_mat=mvar_mat, sc_mat=sc_matrix, group=subj_group,
            )
            if graph is not None:
                mask = (ukf_df["subject"] == subj) & (ukf_df["session"] == sess)
                a_vals = ukf_df[mask].set_index("roi")["a"]
                graph.a_true = torch.tensor(
                    [a_vals.get(roi_meta.iloc[i]["roi_name"], ukf_df["a"].mean())
                     for i in range(len(roi_meta))],
                    dtype=torch.float32,
                )
                empirical_graphs[(subj, sess)] = graph

        if subj not in subjects_list:
            subjects_list.append(subj)

    log.info("MDD graphs: %d (subjects: %d, groups: %s)",
             len(empirical_graphs), len(subjects_list),
             {g: sum(1 for s in subjects_list if groups_map.get(s) == g)
              for g in set(groups_map.values())})
    return empirical_graphs, subjects_list, groups_map


# =============================================================================
# 11 — ANOMALY SCORING
# =============================================================================

def compute_anomaly_scores(gvae_model, graphs, label=""):
    """Compute per-graph and per-ROI anomaly scores (unweighted MSE)."""
    gvae_model.eval()
    scores, roi_errors = [], []
    with torch.no_grad():
        for g in graphs:
            g_c = prepare_graph_for_batching(g) if hasattr(g, "subject") else g
            result = gvae_model(g_c)
            target = g_c.x[:, :3].numpy()
            recon = result["node_recon"].numpy()
            diff = np.clip(target - recon, -1e3, 1e3)
            sq_err = (diff ** 2).mean(axis=1)
            scores.append(sq_err.mean())
            roi_errors.append(sq_err)
    if label:
        log.info("  %s: %d graphs, anomaly = %.6f +/- %.6f",
                 label, len(scores), np.mean(scores), np.std(scores))
    return scores, roi_errors
