import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch
import os
# import wandb

output_dir = "/localssd/chouaib/geo_ai/Model2/SFT/"
os.makedirs(output_dir, exist_ok=True)

merged_model_save_path = "/localssd/chouaib/geo_ai/Model2/SFT/merged_model/"
os.makedirs(merged_model_save_path, exist_ok=True)

peft_config = LoraConfig(
    r=4,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

base_model = AutoModelForCausalLM.from_pretrained(
    "MBZUAI-Paris/Atlas-Chat-2B",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    "MBZUAI-Paris/Atlas-Chat-2B",
    trust_remote_code=True,
    use_fast=False # <--- ADD THIS LINE
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = get_peft_model(base_model, peft_config)

print('This script is the training script of SFT using ATLAS-CHAT-2B.')

model.to(dtype=torch.bfloat16)

def print_trainable_parameters(model):
    trainable_params = 0
    total_params = 0
    for _, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"🧠 Trainable parameters: {trainable_params}")
    print(f"📦 Total parameters: {total_params}")
    print(f"🧪 Trainable %: {100 * trainable_params / total_params:.4f}%")

print_trainable_parameters(model)

training_args = TrainingArguments(
    output_dir=output_dir,
    max_steps=10000,
    per_device_train_batch_size=5,
    per_device_eval_batch_size=5,
    gradient_accumulation_steps=1,
    optim='rmsprop',
    bf16=True,
    logging_steps=100,
    evaluation_strategy="epoch",
    logging_first_step=True,
    learning_rate=1e-5,
    eval_steps=200,
    save_steps=500,
    weight_decay=0.01,
    remove_unused_columns=False,
    warmup_steps=150,
    report_to="wandb",
    run_name="ATLAS_SFT_try",
    logging_dir=os.path.join(output_dir, "logs"),
)

train_df = pd.read_csv('/localssd/chouaib/geo_ai/Work_2/data/parallel_dataset/final_data_splitting/train.csv')
val_df = pd.read_csv('/localssd/chouaib/geo_ai/Work_2/data/parallel_dataset/final_data_splitting/val.csv')

train_df.rename(columns={'darija': 'prompt', 'english': 'chosen'}, inplace=True)
val_df.rename(columns={'darija': 'prompt', 'english': 'chosen'}, inplace=True)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

def tokenize_for_sft(examples):
    source_language = "Moroccan dialect"
    target_language = "English"
    
    full_texts = []
    for i in range(len(examples['prompt'])):
        prompt = examples['prompt'][i]
        chosen_response = examples['chosen'][i]
        
        formatted_example = (
            f"Translate this from [{source_language}] to [{target_language}]:\n"
            f"{source_language}: {prompt}\n"
            f"{target_language}: {chosen_response}"
        )
        full_texts.append(formatted_example)

    MAX_SEQUENCE_LENGTH = 256 

    tokenized_inputs = tokenizer(
        full_texts,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        padding="max_length",
        return_tensors="pt"
    )
    
    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
    }

ds_train = train_dataset.map(tokenize_for_sft, batched=True, num_proc=4, remove_columns=["prompt", "chosen"])
ds_val = val_dataset.map(tokenize_for_sft, batched=True, num_proc=4, remove_columns=["prompt", "chosen"])

print("\nProcessed Training Dataset structure:")
print(ds_train)
print("\nFirst tokenized example (training):")
print(tokenizer.decode(ds_train[0]["input_ids"]))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_val,
    tokenizer=tokenizer,
)

print("\nStarting training...")
trainer.train()
print("\nTraining finished.")

print(f"\nSaving trained model to {output_dir}")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"\nMerging LoRA adapters and saving full model to {merged_model_save_path}")
merged_model = model.merge_and_unload()
merged_model.save_pretrained(merged_model_save_path)
tokenizer.save_pretrained(merged_model_save_path)

print("\nScript execution complete.")