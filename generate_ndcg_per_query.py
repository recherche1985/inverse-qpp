import json
import pytrec_eval

# Load qrels and run file
with open('msmarco-test2019-qrels.txt') as f:
    qrels = pytrec_eval.parse_qrel(f)

with open('bm25_run.json') as f:
    run = json.load(f)

# Compute per-query NDCG scores
evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg'})
ndcg_per_query = {
    qid: metrics['ndcg'] for qid, metrics in evaluator.evaluate(run).items()
}

# Save to JSON
with open('ndcg_per_query.json', 'w') as f:
    json.dump(ndcg_per_query, f, indent=2)

print(f"Saved NDCG scores for {len(ndcg_per_query)} queries.")
