from datasets import Dataset
from peft import LoraConfig
import torch
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, GenerationConfig

from generate_data import generate_dataset
import re
import json

# Load the dataset
dataset = Dataset.from_dict(generate_dataset(1000))

def parse_tags(text: str) -> tuple[str, str] | None:
    pattern = r"^\s*<reasoning>([\s\S]*?)</reasoning>\s*<answer>([\s\S]*?)</answer>\s*$"
    match = re.search(pattern, text.strip())
    
    if match:
        reasoning, answer = match.groups()
        return (reasoning.strip(), answer.strip())
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

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
output_dir = "output/REGENT-T0-1.5B"
run_name = "REGENT-T0-1.5B"

training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type="cosine",
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=16,
    max_prompt_length=256,
    max_completion_length=786,
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=0.1,
    log_on_each_node=False,
    use_vllm=True,
    vllm_device="cuda:7",
    vllm_gpu_memory_utilization=0.7
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=None,
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token if isinstance(tokenizer.eos_token, str) else tokenizer.eos_token[0]

generation_config = GenerationConfig(
    stop_strings=["</answer>"],
    max_new_tokens=1024
)

model.generation_config = generation_config

# # Enable gradient checkpointing before training
# model.config.use_cache = False  # Ensure this is set in the config
# model.gradient_checkpointing_enable()

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[format_reward, reward],
    args=training_args,
    train_dataset=dataset,
    # peft disabled for full runs, useful for testing though
    peft_config=peft_config
)

trainer.train()