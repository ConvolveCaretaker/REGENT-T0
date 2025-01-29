import json

with open("completions.jsonl") as f:
    completions = [json.loads(x) for x in f.readlines()]

for completion in completions:
    print("\n#### PROMPT")
    print(completion["prompts"][-1][0]["content"])
    print("\n#### RESPONSE\n")
    print(completion["completions"][-1][0]["content"])