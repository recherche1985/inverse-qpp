import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
from scipy.stats import kendalltau, spearmanr

# --- Utility functions ---

def reliability_diagram(y_true, y_pred, n_bins=10):
    """Plot reliability diagram for regression."""
    bins = np.linspace(y_true.min(), y_true.max(), n_bins+1)
    bin_indices = np.digitize(y_pred, bins) - 1
    mean_pred = np.array([y_pred[bin_indices==i].mean() if np.any(bin_indices==i) else np.nan for i in range(n_bins)])
    mean_true = np.array([y_true[bin_indices==i].mean() if np.any(bin_indices==i) else np.nan for i in range(n_bins)])
    mae_bin = np.array([np.abs(y_pred[bin_indices==i]-y_true[bin_indices==i]).mean() if np.any(bin_indices==i) else np.nan for i in range(n_bins)])

    plt.figure(figsize=(6,6))
    plt.plot(mean_pred, mean_true, 'o-', label='binned')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', label='ideal')
    plt.xlabel("Mean Predicted")
    plt.ylabel("Mean True")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("MAE per bin:", mae_bin)
    return mae_bin

def residual_plots(y_true, y_pred, X_features=None, feature_names=None):
    residuals = y_true - y_pred
    plt.figure(figsize=(6,4))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title("Residuals vs Predicted")
    plt.grid(True)
    plt.show()

    if X_features is not None:
        for i in range(X_features.shape[1]):
            plt.figure(figsize=(6,4))
            plt.scatter(X_features[:, i], residuals, alpha=0.5)
            plt.axhline(0, color='k', linestyle='--')
            plt.xlabel(feature_names[i] if feature_names else f"Feature {i}")
            plt.ylabel("Residual")
            plt.title(f"Residuals vs Feature {i}")
            plt.grid(True)
            plt.show()

def bootstrap_ci(y_true, y_pred, n_boot=1000, alpha=0.05):
    taus, rhos, rmses = [], [], []
    for _ in range(n_boot):
        idx = resample(np.arange(len(y_true)))
        y_t = y_true[idx]
        y_p = y_pred[idx]
        tau, _ = kendalltau(y_t, y_p)
        rho, _ = spearmanr(y_t, y_p)
        rmse = np.sqrt(np.mean((y_t - y_p)**2))
        taus.append(tau)
        rhos.append(rho)
        rmses.append(rmse)
    ci = lambda arr: (np.percentile(arr, 100*alpha/2), np.percentile(arr, 100*(1-alpha/2)))
    print(f"Kendall tau CI: {ci(taus)}")
    print(f"Spearman rho CI: {ci(rhos)}")
    print(f"RMSE CI: {ci(rmses)}")

# --- Main script ---

# Load JSON files
with open("ndcg_per_query.json", "r") as f:
    ndcg_dict = json.load(f)
    y_true = np.array([ndcg_dict[k] for k in sorted(ndcg_dict.keys())])

with open("inverse_latent_features_ms_marco.json", "r") as f:
    features_dict = json.load(f)
    X_features = np.array([features_dict[k] for k in sorted(features_dict.keys())])

# Generate predicted values (example: add small noise to y_true, replace with actual predictions)
np.random.seed(42)
y_pred = y_true + np.random.normal(scale=0.02, size=y_true.shape)

# Reliability diagram
mae_bins = reliability_diagram(y_true, y_pred, n_bins=10)

# Residual plots
residual_plots(y_true, y_pred, X_features=X_features)

# Bootstrapped confidence intervals
bootstrap_ci(y_true, y_pred, n_boot=1000)
