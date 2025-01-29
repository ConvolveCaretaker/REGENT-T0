from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, GenerationConfig
import torch

from generate_data import generate_dataset
import re
import json
from datetime import datetime

# Load the dataset
dataset = Dataset.from_dict(generate_dataset(1000))

def parse_tags(text: str) -> tuple[str, str] | None:
    match = re.match("<reasoning>(.*)</reasoning>\n<answer>(.*)</answer>", text.strip())
    
    if match:
        return tuple(match.groups())
    else:
        return None

def format_reward(prompts: list[dict], completions: list[dict]) -> float:
    return [1 if parse_tags(x["content"]) else 0 for x in completions]

def reward(prompts: list[str], completions: list[str]):
    # Log completions
    with open("completions.jsonl", "a") as f:
        f.write(json.dumps({"prompts": prompts, "completions": completions}) + "\n")
    
    rewards = []
    for prompt, completion in zip(prompts, completions):
        parsed = parse_tags(completion["content"])
        
        # If tags are malformed, just give it the format reward and continue.
        if not parsed:
            rewards.append(0)
            continue
        
        reasoning, answer = parsed
        
        # The longest the answer could be is 2 characters.
        # It's obviously malformed if it's more than that.
        if len(answer) > 2 or len(answer) == 0:
            rewards.append(0)
            continue
        
        expression, result = re.findall("<equation>(.*) = (.*)</equation>", prompt)
        
        try:
            answer = eval(expression.replace("[?]", answer))
        except:
            rewards.append(0)
            continue
        
        if answer == int(result):
            rewards.append(1)
    
    return rewards

training_args = GRPOConfig(
    output_dir="outputs/Qwen-1.5B-GRPO",
    run_name="Qwen-1.5B-GRPO",
    learning_rate=1e-5,
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=4,
    max_completion_length=1024,
    log_on_each_node=False,
    beta=0.05,
    bf16_full_eval=True,
    bf16=True,
    temperature=0.9,  # Added for generation diversity
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj"],
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct", 
    use_cache=False,
    generation_config=GenerationConfig(stop_strings=["</assistant>"])
)

# Enable gradient checkpointing before training
model.config.use_cache = False  # Ensure this is set in the config
model.gradient_checkpointing_enable()

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward, reward],
    args=training_args,
    train_dataset=dataset,
    # peft disabled for full runs, useful for testing though
    # peft_config=peft_config
)

trainer.train()