import json
import numpy as np
from scipy.stats import entropy

# Load the run file (your retrieval scores)
with open("run.json", "r", encoding="utf-8") as f:
    run_data = json.load(f)

retrieval_qpp_features = {}

for qid, doc_scores in run_data.items():
    scores = list(doc_scores.values())
    if not scores:
        continue

    scores_np = np.array(scores)
    norm_scores = scores_np / np.sum(scores_np)

    features = {
        "max_score": float(np.max(scores_np)),
        "min_score": float(np.min(scores_np)),
        "mean_score": float(np.mean(scores_np)),
        "score_variance": float(np.var(scores_np)),
        "score_stddev": float(np.std(scores_np)),
        "score_entropy": float(entropy(norm_scores, base=2))  # entropy in bits
    }

    retrieval_qpp_features[qid] = features

# Save the features
with open("retrieval_qpp_features.json", "w", encoding="utf-8") as f:
    json.dump(retrieval_qpp_features, f, indent=2, ensure_ascii=False)

print("✅ Retrieval-based QPP features saved to retrieval_qpp_features.json")
