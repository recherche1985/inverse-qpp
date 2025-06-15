import json
from collections import defaultdict

input_file = 'run.robust04.txt'
output_file = 'run.json'

run = defaultdict(dict)

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        qid, _, docid, rank, score, _ = line.strip().split()
        run[qid][docid] = float(score)

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(run, f, indent=2)

print(f"✅ Converted {input_file} to {output_file}")
