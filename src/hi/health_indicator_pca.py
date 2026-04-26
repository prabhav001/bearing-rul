"""
src/health_indicator.py
=======================
Constructs a Health Indicator (HI) for IMS bearings using
PCA-based T² statistic following Wang (2012).

Methodology:
    1. Load smoothed feature matrix from features.py output
    2. Normalize using healthy window statistics
    3. Fit PCA on healthy window (snapshots 0 to FPT)
    4. Compute T² statistic for all snapshots
    5. Plot and save HI

Reference:
    Wang, T. (2012). Bearing Life Prediction Based on Vibration
    Signals: A Case Study and Lessons Learned. IEEE PHM.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# test = "1st_test"
# bearing = "B2"
# # ── Paths ─────────────────────────────────────────────────────
# FEATURES_PATH = f"./data/processed/features_ims_{test}_{bearing}.csv"
# HI_PATH       = f"./data/processed/hi_ims_{test}_{bearing}.csv"

# ── Healthy window ────────────────────────────────────────────
FPT_IDX       = 530     # from EDA — first prediction time

# ── PCA variance threshold ────────────────────────────────────
VARIANCE_THRESHOLD = 0.95    # keep components explaining 95% variance

# ── Fault detection threshold ─────────────────────────────────
T2_THRESHOLD  = 50      # Wang's value — we'll adjust after seeing the plot

def normalize(X_healthy, X_all):
    """
    Normalize all data using healthy window statistics.

    Why healthy only:
        If we compute mean/std from the full run (including degraded
        snapshots), the statistics are contaminated by fault behavior.
        The normalization would shift the healthy baseline away from
        zero, making T² less sensitive to early degradation.

    Args:
        X_healthy : (FPT, n_features)  — healthy snapshots only
        X_all     : (n_snapshots, n_features) — all snapshots

    Returns:
        Z_healthy : normalized healthy data
        Z_all     : normalized full data (using healthy stats)
        mu        : mean of each feature (from healthy)
        sigma     : std of each feature (from healthy)
    """
    mu    = X_healthy.mean(axis=0)      # shape (n_features,)
    sigma = X_healthy.std(axis=0)       # shape (n_features,)

    # avoid division by zero for constant features
    sigma[sigma == 0] = 1.0

    Z_healthy = (X_healthy - mu) / sigma
    Z_all     = (X_all     - mu) / sigma

    return Z_healthy, Z_all, mu, sigma

def fit_pca(Z_healthy, variance_threshold=VARIANCE_THRESHOLD):
    """
    Fit PCA on healthy normalized data following Wang (2012).

    Steps:
        1. Compute covariance matrix of healthy data
        2. Eigen decomposition
        3. Sort eigenvalues descending
        4. Select top l components explaining variance_threshold variance
        5. Return eigenvectors and eigenvalues of selected subspace

    Args:
        Z_healthy          : normalized healthy data (FPT, n_features)
        variance_threshold : cumulative variance to retain (0.95)

    Returns:
        V_l      : selected eigenvectors (n_features, l)
        Lambda_l : diagonal eigenvalue matrix (l, l)
        l        : number of components selected
        cum_var  : full cumulative variance array (for plotting)
    """
    n = len(Z_healthy)

    # step 1 — covariance matrix
    # shape (n_features, n_features)
    # (1/n) × Z.T @ Z  gives covariance since Z is already mean-centered
    # (normalization step already removed the mean)
    C = (1 / n) * Z_healthy.T @ Z_healthy

    # step 2 — eigen decomposition
    # np.linalg.eigh is used instead of eig because:
    # - C is symmetric (covariance matrix always is)
    # - eigh is faster and numerically more stable for symmetric matrices
    # - eigh guarantees real eigenvalues (eig may return complex for
    #   symmetric matrices due to floating point errors)
    eigenvalues, eigenvectors = np.linalg.eigh(C)

    # step 3 — sort descending
    # eigh returns eigenvalues in ascending order — we need descending
    # so PC1 (most variance) comes first
    idx          = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # step 4 — select top l components
    # cumulative variance = how much variance is explained by
    # PC1, PC1+PC2, PC1+PC2+PC3, ...
    total_var = np.sum(eigenvalues)
    cum_var   = np.cumsum(eigenvalues) / total_var

    # find smallest l such that cumulative variance >= threshold
    # searchsorted finds insertion point — +1 because we need AT LEAST
    # threshold variance, not just below it
    l = int(np.searchsorted(cum_var, variance_threshold)) + 1

    # clip to valid range
    l = min(l, len(eigenvalues))

    print(f"  PCA: selected {l} components explaining "
          f"{cum_var[l-1]*100:.1f}% variance")
    print(f"  Eigenvalues: {eigenvalues[:l].round(4)}")

    # step 5 — extract selected subspace
    V_l      = eigenvectors[:, :l]          # shape (n_features, l)
    Lambda_l = np.diag(eigenvalues[:l])     # shape (l, l)

    return V_l, Lambda_l, l, cum_var

def compute_t2(Z_all, V_l, Lambda_l):
    """
    Compute Hotelling's T² statistic for every snapshot.

    Formula (Wang 2012):
        T²(t) = z_t @ V_l @ Lambda_l⁻¹ @ V_l.T @ z_t

    Intuition:
        T² measures the Mahalanobis distance of each snapshot
        from the healthy PCA subspace, scaled by eigenvalues.

        - Healthy snapshot  → lies within healthy subspace → T² small
        - Degraded snapshot → drifts outside subspace     → T² large

    Args:
        Z_all    : normalized full data (n_snapshots, n_features)
        V_l      : selected eigenvectors (n_features, l)
        Lambda_l : diagonal eigenvalue matrix (l, l)

    Returns:
        t2 : T² values (n_snapshots,)
    """
    # invert the eigenvalue diagonal matrix
    # Lambda_l⁻¹ simply inverts each diagonal element
    # i.e. 1/λ₁, 1/λ₂, ..., 1/λ_l
    Lambda_inv = np.diag(1.0 / np.diag(Lambda_l))

    # precompute the projection matrix
    # shape: (n_features, n_features)
    # this is computed once and reused for all snapshots
    # P = V_l @ Lambda_inv @ V_l.T
    P = V_l @ Lambda_inv @ V_l.T

    # compute T² for each snapshot
    # z_t @ P @ z_t  is a scalar for each snapshot t
    # vectorized across all snapshots at once:
    # (n_snapshots, n_features) @ (n_features, n_features) → (n_snapshots, n_features)
    # then element-wise multiply with Z_all and sum → (n_snapshots,)
    t2 = np.sum(Z_all @ P * Z_all, axis=1)

    return t2
def plot_hi(test, bearing, t2, fpt_idx=FPT_IDX, threshold=T2_THRESHOLD):
    """
    Plot T² Health Indicator over time.

    Args:
        t2        : T² values (n_snapshots,)
        fpt_idx   : first prediction time index
        threshold : fault detection threshold
        test      : test folder name
        bearing   : bearing name
    """
    plt.figure(figsize=(14, 4))
    plt.plot(t2, linewidth=0.8, color="steelblue", label="T² HI")
    plt.yscale("log")
    plt.axvline(x=fpt_idx,   color="orange", linewidth=1.2,
                linestyle="--", label=f"FPT ≈ {fpt_idx}")
    #plt.axvline(x=980,       color="red",    linewidth=1.2, linestyle="--", label="Failure ≈ 960")
    plt.axhline(y=50, color="green",  linewidth=1.2,
                linestyle="--", label=f"Threshold = {threshold}")
    plt.title(f"PCA-based T² Health Indicator — {test}, Bearing {bearing}")
    plt.xlabel("Snapshot index")
    plt.ylabel("T²")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_eigenvalues(cum_var):
    """
    Plot cumulative variance explained by principal components.
    Helps visualize how many components are needed.

    Args:
        cum_var : cumulative variance array from fit_pca
    """
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, len(cum_var)+1), cum_var * 100,
             marker="o", linewidth=1.2, color="steelblue")
    plt.axhline(y=95, color="red", linewidth=1,
                linestyle="--", label="95% threshold")
    plt.title("Cumulative Variance Explained by Principal Components")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative variance (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def save_hi(t2, output_path):
    """
    Save T² Health Indicator to CSV.

    Args:
        t2          : T² values (n_snapshots,)
        output_path : path to save CSV
    """
    df = pd.DataFrame({"snapshot": np.arange(len(t2)), "T2": t2})
    df.to_csv(output_path, index=False)
    print(f"  HI saved to {output_path}")


def run(test, bearing):
    """
    Full HI construction pipeline:
        load features → normalize → fit PCA → compute T² → plot → save

    Args:
        features_path : path to smoothed feature CSV
        output_path   : path to save HI CSV
    """
    features_path = f"./data/processed/features_ims_{test}_{bearing}.csv"
    output_path   = f"./data/processed/hi_ims_{test}_{bearing}.csv"
    # ── Step 1 — Load features ────────────────────────────────
    print(f"\nLoading features from {features_path}")
    df   = pd.read_csv(features_path)
    X    = df.values.astype(np.float64)    # shape (984, 18)
    print(f"  feature matrix shape : {X.shape}")

    # ── Step 2 — Split healthy and full ───────────────────────
    X_healthy = X[:FPT_IDX, :]             # shape (530, 18)
    X_all     = X                          # shape (984, 18)
    print(f"  healthy window       : snapshots 0 to {FPT_IDX}")
    print(f"  full window          : snapshots 0 to {len(X)}")

    # ── Step 3 — Normalize ────────────────────────────────────
    print(f"\nNormalizing using healthy window statistics...")
    Z_healthy, Z_all, mu, sigma = normalize(X_healthy, X_all)

    # ── Step 4 — Fit PCA on healthy data ──────────────────────
    print(f"\nFitting PCA on healthy window (variance threshold={VARIANCE_THRESHOLD})...")
    V_l, Lambda_l, l, cum_var = fit_pca(Z_healthy)

    # ── Step 5 — Plot eigenvalue curve ────────────────────────
    plot_eigenvalues(cum_var)

    # ── Step 6 — Compute T² for all snapshots ─────────────────
    print(f"\nComputing T² for all {len(X_all)} snapshots...")
    t2 = compute_t2(Z_all, V_l, Lambda_l)
    print(f"  T² min : {t2.min():.4f}")
    print(f"  T² max : {t2.max():.4f}")
    print(f"  T² mean (healthy window) : {t2[:FPT_IDX].mean():.4f}")
    print(f"  T² mean (degraded window): {t2[FPT_IDX:960].mean():.4f}")

    # ── Step 7 — Plot HI ──────────────────────────────────────
    plot_hi(test=test, bearing=bearing, t2=t2)

    # ── Step 8 — Save ─────────────────────────────────────────
    save_hi(t2, output_path)

    return t2


if __name__ == "__main__":
    run(
        test="1st_test",
        bearing="B4"
    )