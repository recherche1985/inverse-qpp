# compute_oracle_upper_bound.py
import json, os, argparse, pickle
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr, kendalltau

def load_json(path):
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)

def metrics(y_true, y_pred):
    pear = pearsonr(y_true, y_pred)[0] if len(y_true)>1 else float("nan")
    spear = spearmanr(y_true, y_pred)[0] if len(y_true)>1 else float("nan")
    kend = kendalltau(y_true, y_pred)[0] if len(y_true)>1 else float("nan")
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return {"pearson": pear, "spearman": spear, "kendall": kend, "mae": mae, "rmse": rmse}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="retrieval_qpp_features.json")
    parser.add_argument("--ndcg", default="ndcg_per_query.json")
    parser.add_argument("--inverse", default="inverse_qpp_predictions.json", help="predicted latent x (optional)")
    parser.add_argument("--out_json", default="oracle_results.json")
    parser.add_argument("--out_tex", default="oracle_table.tex")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    feat_path = args.features
    ndcg_path = args.ndcg
    inv_path = args.inverse

    assert os.path.exists(feat_path), f"Missing {feat_path}"
    assert os.path.exists(ndcg_path), f"Missing {ndcg_path}"

    feats = load_json(feat_path)   # dict: qid -> {feature_name: value, ...}
    ndcg = load_json(ndcg_path)    # dict: qid -> ndcg value

    # Build aligned lists of queries which have both features and ndcg
    common_qids = [qid for qid in feats.keys() if qid in ndcg]
    common_qids.sort()
    print(f"Using {len(common_qids)} queries (aligned features & ndcg).")

    # Build X matrix (keep a consistent feature order)
    sample_q = common_qids[0]
    feat_keys = sorted(feats[sample_q].keys())
    X = np.array([[feats[qid].get(k, 0.0) for k in feat_keys] for qid in common_qids])
    y = np.array([ndcg[qid] for qid in common_qids], dtype=float)

    # optionally load reconstructed inverse latent features (x_hat)
    inverse_latent = None
    if os.path.exists(inv_path):
        inv_json = load_json(inv_path)
        # expect inv_json: qid -> list or dict; handle both
        inv_items = {}
        for qid in common_qids:
            if qid in inv_json:
                val = inv_json[qid]
                if isinstance(val, dict):
                    # same keys -> extract in feat_keys order
                    inv_items[qid] = np.array([val.get(k, 0.0) for k in feat_keys])
                else:
                    # assume list in same order
                    inv_items[qid] = np.array(val)
        if len(inv_items) > 0:
            inverse_latent = np.array([inv_items[qid] for qid in common_qids])
            print(f"Loaded reconstructed latent features for {inverse_latent.shape[0]} queries.")
        else:
            print("No reconstructed latent features matched common qids in inverse file; skipping inverse comparison.")

    # Split
    X_train, X_test, y_train, y_test, q_train, q_test = train_test_split(
        X, y, common_qids, test_size=0.2, random_state=args.seed
    )

    # Standardize X for GBRT? GBRT doesn't require scaling but keep as option (we won't scale X here)
    forward_model_path = "forward_model.pkl"
    if os.path.exists(forward_model_path):
        print("Loading saved forward model:", forward_model_path)
        with open(forward_model_path, "rb") as f:
            forward = pickle.load(f)
    else:
        print("Training forward GBRT on true latent features...")
        forward = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=args.seed)
        forward.fit(X_train, y_train)
        with open(forward_model_path, "wb") as f:
            pickle.dump(forward, f)
        print("Saved forward model to", forward_model_path)

    # Oracle predictions: use true X_test as input to forward model
    y_oracle = forward.predict(X_test)

    # If inverse_latent available, compute predictions from reconstructed X_hat
    y_inverse = None
    if inverse_latent is not None:
        # align inverse_latent to test set indices (we built inverse_latent over common_qids order)
        # create mask for test positions
        idxs = [common_qids.index(qid) for qid in q_test]
        X_hat_test = inverse_latent[idxs]
        # ensure size compatibility
        if X_hat_test.shape[1] != X_test.shape[1]:
            print("Warning: reconstructed x dimensionality differs from true x. Attempting to adapt/truncate.")
            min_dim = min(X_hat_test.shape[1], X_test.shape[1])
            X_hat_test = X_hat_test[:, :min_dim]
            X_test_trunc = X_test[:, :min_dim]
            y_oracle = forward.predict(X_test_trunc)
            y_inverse = forward.predict(X_hat_test)
        else:
            y_inverse = forward.predict(X_hat_test)

    # also produce forward-from-true-x (for clarity, same as oracle)
    y_forward_from_x = y_oracle

    # compute metrics
    res = {}
    res["oracle"] = metrics(y_test, y_oracle)
    if y_inverse is not None:
        res["inverse_pipeline"] = metrics(y_test, y_inverse)
    else:
        res["inverse_pipeline"] = None

    print("\n=== RESULTS ===")
    print("Oracle (F(x)):", res["oracle"])
    if res["inverse_pipeline"]:
        print("Inverse pipeline (F(x_hat)):", res["inverse_pipeline"])
        # compute gap
        gap = {k: res["oracle"][k] - res["inverse_pipeline"][k] if res["inverse_pipeline"] else None for k in res["oracle"].keys()}
        print("Oracle - Inverse (positive means oracle better):", gap)

    # save JSON
    with open(args.out_json, "w", encoding="utf8") as f:
        json.dump({"q_test": q_test, "results": res}, f, indent=2)

    # write LaTeX table
    with open(args.out_tex, "w", encoding="utf8") as f:
        f.write("\\begin{table}[ht]\n\\centering\n")
        f.write("\\caption{Oracle-inverse upper bound vs. inverse pipeline.}\n")
        f.write("\\label{tab:oracle_upper}\n")
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("Model & Pearson $r$ & Spearman $\\rho$ & Kendall $\\tau$ & RMSE \\\\\n\\midrule\n")
        f.write(f"Oracle (forward on true $x$) & {res['oracle']['pearson']:.3f} & {res['oracle']['spearman']:.3f} & {res['oracle']['kendall']:.3f} & {res['oracle']['rmse']:.3f} \\\\\n")
        if res["inverse_pipeline"]:
            f.write(f"Inverse pipeline (forward on $\\hat{{x}}$) & {res['inverse_pipeline']['pearson']:.3f} & {res['inverse_pipeline']['spearman']:.3f} & {res['inverse_pipeline']['kendall']:.3f} & {res['inverse_pipeline']['rmse']:.3f} \\\\\n")
        else:
            f.write("Inverse pipeline (forward on $\\hat{x}$) & --- & --- & --- & --- \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print("Wrote", args.out_tex, "and", args.out_json)

if __name__ == "__main__":
    main()
