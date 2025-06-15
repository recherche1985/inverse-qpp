import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import kendalltau, spearmanr
import matplotlib.pyplot as plt

# === Load QPP Features and NDCG Labels ===
with open("retrieval_qpp_features.json", "r") as f:
    features_dict = json.load(f)

with open("ndcg_per_query.json", "r") as f:
    ndcg_dict = json.load(f)

# === Prepare X (features), y (observed metric), qids ===
query_ids = [qid for qid in features_dict if qid in ndcg_dict]
X = np.array([[features_dict[qid][k] for k in sorted(features_dict[qid])] for qid in query_ids])
y = np.array([ndcg_dict[qid] for qid in query_ids])

# === Normalize Features and Target ===
scaler_X = StandardScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# === Split Data ===
X_train, X_test, y_train, y_test, qids_train, qids_test = train_test_split(
    X_scaled, y_scaled, query_ids, test_size=0.2, random_state=42
)

# === Forward Model: Learn F(x) ≈ y ===
forward_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
forward_model.fit(X_train, y_train)
y_pred_forward = forward_model.predict(X_test)

print("✅ FORWARD MODEL: Predicting y from x")
print("  Kendall Tau:", round(kendalltau(y_test, y_pred_forward).statistic, 4))
print("  Spearman Correlation:", round(spearmanr(y_test, y_pred_forward).statistic, 4))
print("  RMSE:", round(mean_squared_error(y_test, y_pred_forward, squared=False), 4))

# === Inverse Model: Learn F⁻¹(y) ≈ x ===
inverse_models = []
X_inverse_train = X_train
X_inverse_test = X_test

x_pred_components = []

for dim in range(X_inverse_train.shape[1]):
    inv_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=dim)
    inv_model.fit(y_train.reshape(-1, 1), X_inverse_train[:, dim])
    x_pred_dim = inv_model.predict(y_test.reshape(-1, 1))
    x_pred_components.append(x_pred_dim)
    inverse_models.append(inv_model)

x_pred_inverse = np.stack(x_pred_components, axis=1)

# === Validate Inverse Recovery via Reconstruction (F(x̂) ≈ y) ===
y_reconstructed = forward_model.predict(x_pred_inverse)

print("✅ INVERSE MODEL: Reconstructing y from x̂ = F⁻¹(y)")
print("  Kendall Tau:", round(kendalltau(y_test, y_reconstructed).statistic, 4))
print("  Spearman Correlation:", round(spearmanr(y_test, y_reconstructed).statistic, 4))
print("  RMSE:", round(mean_squared_error(y_test, y_reconstructed, squared=False), 4))

# === Save Outputs ===
with open("inverse_latent_features.json", "w") as f:
    json.dump({qid: x.tolist() for qid, x in zip(qids_test, x_pred_inverse)}, f, indent=2)

with open("reconstructed_ndcg.json", "w") as f:
    json.dump(dict(zip(qids_test, y_reconstructed.tolist())), f, indent=2)


# === Plot Inverse Model Predictions vs True NDCG ===

# Inverse transform scaled data back to original scale
y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
y_reconstructed_orig = scaler_y.inverse_transform(y_reconstructed.reshape(-1, 1)).ravel()

plt.figure(figsize=(8, 6))
plt.scatter(y_test_orig, y_reconstructed_orig, alpha=0.6, edgecolors='b', s=40)
plt.plot([min(y_test_orig), max(y_test_orig)], [min(y_test_orig), max(y_test_orig)], 'r--', linewidth=2, label='Ideal Prediction')
plt.xlabel('True NDCG')
plt.ylabel('Predicted NDCG (Inverse Model)')
plt.title('Inverse Model Predictions vs True NDCG on MS MARCO')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("inverse_model_msmarco_ranking.png", dpi=300)
plt.show()
