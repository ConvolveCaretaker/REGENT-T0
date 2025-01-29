import json

with open("completions.jsonl") as f:
    completions = [json.loads(x) for x in f.readlines()]

print(completions[-1])