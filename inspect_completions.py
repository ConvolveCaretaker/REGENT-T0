import json

with open("completions.jsonl") as f:
    completions = [json.loads(x) for x in f.readlines()]

print(completions[-1]["prompts"][-1][0]["content"])
print("\n#### RESPONSE\n")
print(completions[-1]["completions"][-1][0]["content"])