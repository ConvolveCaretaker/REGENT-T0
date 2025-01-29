from dotenv import load_dotenv
from openai import OpenAI
import os

import random

valid_ops = ["+", "-", "*"]  # we don't use division because that introduces decimals

"""
Generates a single valid logic puzzle problem.
"""
def generate_problem(length: int) -> tuple[str, str]:
    numbers = [random.randrange(1, 99) for _ in range(length)]
    ops = [random.choice(valid_ops) for _ in range(length - 1)]

    equation = str(numbers[0])
    for num, op in zip(numbers[1:], ops):
        equation += f" {op} {num}"
    
    result = eval(equation)

    if random.randint(0, 1) == 0:
        index = random.randint(0, len(ops)-1)
        
        answer = ops[index]
        ops[index] = f"(?)"
    else:
        index = random.randint(0, len(numbers)-1)
        
        answer = numbers[index]
        numbers[index] = f"(?)"
    
    changed = str(numbers[0])
    for num, op in zip(numbers[1:], ops):
        changed += f" {op} {num}"
    
    changed += f" = {result}"

    return changed, str(answer)

SYSTEM_PROMPT = """After this message, you will be asked a question. When answering the question, follow this format:
<reasoning>
[Work through the problem, before you are satisfied and are ready to answer]
</reasoning>
<answer>
[Give the final answer here.]
</answer>"""

Q_PROMPT = """Below is an equation. Either a single operator or a single number has been replaced with "[?]".
Your goal is to find the number or operator that has been replaced.
The possible operators are +, -, and *.
The answer should consist only of the missing operator or number.

{}"""

def generate_dataset(n: int) -> dict:
    data = {"prompt": [], "completion": [], "ground_truth": []}

    for _ in range(n):
        question, answer = generate_problem(random.randint(2, 30))

        prompt = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": Q_PROMPT.format(question)}
            ]

        data["prompt"].append(prompt)
        data["completion"].append([{"role": "assistant", "content": answer}])
        data["ground_truth"].append(answer)
    
    return data

if __name__ == "__main__":
    print("Test code, this file is not used for training.")
    print(generate_dataset(10))