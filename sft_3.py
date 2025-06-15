import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch
import os

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
    use_fast=False
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.bos_token is None:
    tokenizer.add_special_tokens({'bos_token': '<s>'})
    # Resize model embeddings to account for new tokens
    base_model.resize_token_embeddings(len(tokenizer))

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
    remove_unused_columns=True,
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
    MAX_SEQUENCE_LENGTH = 256

    all_input_ids = []
    all_attention_mask = []
    all_labels = []

    for i in range(len(examples['prompt'])):
        prompt = examples['prompt'][i]
        chosen_response = examples['chosen'][i]
        
        prompt_text = (
            f"Translate this from [{source_language}] to [{target_language}]:\n"
            f"{source_language}: {prompt}\n"
            f"{target_language}: "
        )
        
        tokenized_prompt = tokenizer(prompt_text, add_special_tokens=True, truncation=False)
        tokenized_response = tokenizer(chosen_response, add_special_tokens=False, truncation=False)

        full_input_ids = tokenized_prompt['input_ids'] + tokenized_response['input_ids'] + [tokenizer.eos_token_id]
        full_attention_mask = tokenized_prompt['attention_mask'] + tokenized_response['attention_mask'] + [1]

        labels = [-100] * len(tokenized_prompt['input_ids']) + full_input_ids[len(tokenized_prompt['input_ids']):]

        if len(full_input_ids) > MAX_SEQUENCE_LENGTH:
            full_input_ids = full_input_ids[:MAX_SEQUENCE_LENGTH]
            full_attention_mask = full_attention_mask[:MAX_SEQUENCE_LENGTH]
            labels = labels[:MAX_SEQUENCE_LENGTH]
        
        all_input_ids.append(full_input_ids)
        all_attention_mask.append(full_attention_mask)
        all_labels.append(labels)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    }

ds_train = train_dataset.map(
    tokenize_for_sft, 
    batched=True, 
    num_proc=4, 
    remove_columns=[col for col in train_dataset.column_names if col not in ['input_ids', 'attention_mask', 'labels']]
)
ds_val = val_dataset.map(
    tokenize_for_sft, 
    batched=True, 
    num_proc=4, 
    remove_columns=[col for col in val_dataset.column_names if col not in ['input_ids', 'attention_mask', 'labels']]
)

print("\nProcessed Training Dataset structure:")
print(ds_train)
print("\nFirst tokenized example (training):")
print(tokenizer.decode(ds_train[0]["input_ids"]))
print("\nFirst tokenized example (training) labels:")
decoded_labels = [token_id for token_id in ds_train[0]["labels"] if token_id != -100]
print(tokenizer.decode(decoded_labels))

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