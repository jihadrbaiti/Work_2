# -*- coding: utf-8 -*-
import torch
import pandas as pd
import wandb
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import CPOConfig, CPOTrainer
from peft import LoraConfig, get_peft_model

# ------------------------------
# Load JAIS-13B model and tokenizer
# ------------------------------
model_path = "core42/jais-13b"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# ------------------------------
# Apply LoRA
# ------------------------------
peft_config = LoraConfig(
    r=4,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(base_model, peft_config)

# ------------------------------
# W&B Setup
# ------------------------------
wandb.login(key='86570a60523435fb4d496c0e63e8ae11c308bae2')
print("🚀 Training JAIS-13B with CPO...")

# ------------------------------
# CPO Config
# ------------------------------
cpo_config = CPOConfig(
    output_dir='/localssd/chouaib/geo_ai/Model2/Jais_CPO',
    logging_dir='/localssd/chouaib/geo_ai/Model2/Jais_CPO/logs',
    run_name='JAIS_CPO',
    max_steps=10000,
    eval_strategy="epoch",
    eval_steps=200,
    generate_during_eval=True,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=150,
    weight_decay=0.01,
    optim='rmsprop',
    logging_steps=100,
    logging_first_step=True,
    report_to='wandb',
    bf16=True,
    max_length=256,
    max_prompt_length=256,
    max_completion_length=256,
    remove_unused_columns=False,
    beta=0.2,
    cpo_alpha=0.8
)

# ------------------------------
# Load Data
# ------------------------------
train = pd.read_csv('/localssd/chouaib/geo_ai/Work_2/data/parallel_dataset/final_data_splitting/train.csv')
val = pd.read_csv('/localssd/chouaib/geo_ai/Work_2/data/parallel_dataset/final_data_splitting/val.csv')

train.rename(columns={'darija': 'prompt', 'english': 'chosen', 'perturbation': 'rejected'}, inplace=True)
val.rename(columns={'darija': 'prompt', 'english': 'chosen', 'perturbation': 'rejected'}, inplace=True)

train = Dataset.from_pandas(train)
val = Dataset.from_pandas(val)

# ------------------------------
# Preprocess: no chat formatting
# ------------------------------
def process(row):
    source_language = "Moroccan dialect"
    target_language = "English"
    prompt = f"Translate this from [{source_language}] to [{target_language}]:\n{source_language}: {row['prompt']}\n{target_language}: "
    row["chosen"] = prompt + row["chosen"]
    row["rejected"] = prompt + row["rejected"]
    return row

ds_train = train.map(process, batch_size=200, num_proc=4)
ds_val = val.map(process, batch_size=200, num_proc=4)

# ------------------------------
# Set pad_token if missing
# ------------------------------
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ------------------------------
# Log trainable parameters
# ------------------------------
def print_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Trainable parameters: {trainable_params}")
    print(f"📦 Total parameters: {total_params}")
    print(f"🧪 Trainable %: {100 * trainable_params / total_params:.4f}%")

print_trainable_parameters(model)

# ------------------------------
# Start Training
# ------------------------------
trainer = CPOTrainer(
    model=model,
    args=cpo_config,
    train_dataset=ds_train,
    eval_dataset=ds_val,
    processing_class=tokenizer,
    peft_config=peft_config,
)

trainer.train()

# ------------------------------
# Save model + tokenizer
# ------------------------------
trainer.save_model(cpo_config.output_dir)
tokenizer.save_pretrained(cpo_config.output_dir)
