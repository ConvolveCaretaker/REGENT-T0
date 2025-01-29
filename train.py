from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, GenerationConfig

from generate_data import generate_dataset
import re
import json

# Load the dataset
dataset = Dataset.from_dict(generate_dataset(1000))

def parse_tags(text: str) -> tuple[str, str] | None:
    match = re.match("<reasoning>(.*)</reasoning>\n<answer>(.*)</answer>", text.strip())
    
    if match:
        return tuple(match.groups())
    else:
        return None

def format_reward(prompts: list[dict], completions: list[dict], ground_truth: list[str]) -> float:
    return [1 if parse_tags(x[0]["content"]) else 0 for x in completions]

def reward(prompts: list[dict], completions: list[dict], ground_truth: list[str]):
    # Log completions
    with open("completions.jsonl", "a") as f:
        f.write(json.dumps({"prompts": prompts, "completions": completions, "ground_truths": ground_truth}) + "\n")
    
    rewards = []
    for prompt, completion, truth in zip(prompts, completions, ground_truth):
        parsed = parse_tags(completion[0]["content"])
        
        # If tags are malformed, just give it the format reward and continue.
        if not parsed:
            rewards.append(0)
            continue
        
        _, answer = parsed
        
        if answer.strip() == truth:
            rewards.append(1)
        else:
            rewards.append(0)
    
    return rewards

training_args = GRPOConfig(
    output_dir="outputs/Llama-3.2-3B-GRPO",
    run_name="Llama-3.2-3B-GRPO",
    learning_rate=1e-5,
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=8,
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
    "meta-llama/Llama-3.2-3B-GRPO", 
    use_cache=False,
)

generation_config = GenerationConfig(
    stop_strings=["</answer>"],
    max_new_tokens=1024,
    pad_token_id=model.config.eos_token_id,
    eos_token_id=model.config.eos_token_id,
)

model.generation_config = generation_config

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