import json
import pytrec_eval

# Load qrels
qrels = {}
with open("qrels.txt", "r", encoding="utf-8") as f:
    for line in f:
        qid, _, docid, rel = line.strip().split()
        qrels.setdefault(qid, {})[docid] = int(rel)

# Load run
with open("run.json", "r", encoding="utf-8") as f:
    run = json.load(f)

# Initialize evaluator
evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg'})

# Compute per-query NDCG
ndcg_per_query = {}
for qid in run:
    try:
        results = evaluator.evaluate({qid: run[qid]})
        ndcg_per_query[qid] = results[qid]['ndcg']
    except Exception:
        ndcg_per_query[qid] = 0.0

# Save to JSON
with open("ndcg_per_query.json", "w", encoding="utf-8") as f:
    json.dump(ndcg_per_query, f, indent=2)

print("✅ Saved NDCG scores per query to ndcg_per_query.json")
