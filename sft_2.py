import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch
import os
import wandb

# ====== Paths ======
train_path = '/localssd/chouaib/geo_ai/Work_2/data/parallel_dataset/final_data_splitting/train.csv'
val_path = '/localssd/chouaib/geo_ai/Work_2/data/parallel_dataset/final_data_splitting/val.csv'
output_dir = "/localssd/chouaib/geo_ai/Model2/SFT/"
merged_model_save_path = os.path.join(output_dir, "merged_model_final")
#wandb.init(id='3byyd8fc', resume='must', project="huggingface", name="ATLAS_SFT_2" )

# ====== Load and Prepare Dataset ======
def load_and_prepare_data(train_path, val_path):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # Rename columns
    train_df.rename(columns={'darija': 'prompt', 'english': 'response'}, inplace=True)
    val_df.rename(columns={'darija': 'prompt', 'english': 'response'}, inplace=True)

    # Define chat format
    def format_chat(row):
        source_language = "Moroccan dialect"
        target_language = "English"
        prompt = f"Translate this from [Moroccan dialect] to [English]:\nMoroccan dialect: {row['prompt']}\nEnglish:"
        system = f'You are a native speaker of both [{source_language}] and [{target_language}]. You are an expert post editor of translations from [{source_language}] into [{target_language}] and a helpful assistant dedicated to improving translation quality. You will be provided with a source sentence in [{source_language}] and its translation in [{target_language}]. Your task is to carefully analyze the provided source sentence and translation, and suggest improvements to the translation. Note that you only need to generate a refined translation in [{target_language}] and do not generate anything else.'

        return {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": row["response"]}
            ]
        }

    train_ds = Dataset.from_pandas(train_df).map(format_chat)
    val_ds = Dataset.from_pandas(val_df).map(format_chat)

    return train_ds, val_ds

# ====== Load Model and Tokenizer ======
model_id = "MBZUAI-Paris/Atlas-Chat-2B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Ensure chat template is defined
tokenizer.chat_template = """{% for message in messages %}
{% if message['role'] == 'system' %}<|system|>
{{ message['content'] }}
<|end|>
{% elif message['role'] == 'user' %}<|user|>
{{ message['content'] }}
<|end|>
{% elif message['role'] == 'assistant' %}<|assistant|>
{{ message['content'] }}
<|end|>
{% endif %}
{% endfor %}"""


special_tokens = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
tokenizer.add_special_tokens({
    "additional_special_tokens": special_tokens
})
# ====== Add special tokens ======

# ====== Tokenize ======
def tokenize(sample):
    input_ids = tokenizer.apply_chat_template(
        sample["messages"],
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256
    )[0]
    return {"input_ids": input_ids, "labels": input_ids.clone()}

# ====== Load and Tokenize Dataset ======
train_ds, val_ds = load_and_prepare_data(train_path, val_path)
tokenized_train = train_ds.map(tokenize, remove_columns=train_ds.column_names)
tokenized_val = val_ds.map(tokenize, remove_columns=val_ds.column_names)

if tokenizer.chat_template is None:
    from trl import SIMPLE_SFT_CHAT_TEMPLATE
    tokenizer.chat_template = SIMPLE_SFT_CHAT_TEMPLATE

# ====== Model with LoRA ======
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    use_cache=False
)
base_model.resize_token_embeddings(len(tokenizer))

peft_config = LoraConfig(
    r=4,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, peft_config)

# ====== Trainer Config ======
training_args = TrainingArguments(
    output_dir=output_dir,
    max_steps=10000,
    eval_strategy="epoch",
    per_device_train_batch_size=5,
    per_device_eval_batch_size=5,
    gradient_accumulation_steps=1,
    optim='rmsprop',
    bf16=True,
    logging_steps=100,
    evaluation_strategy="epoch",
    logging_first_step=True,
    learning_rate=1e-5,
    eval_steps = 200,
    save_steps = 500,
    weight_decay=0.01,
    remove_unused_columns=False,
    warmup_steps=150,
    report_to="wandb",  # Optional: set to "none" if not using WandB
    run_name="ATLAS_SFT_2",
    logging_dir="/localssd/chouaib/geo_ai/Model2/SFT/logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
)

# ====== Train ======
trainer.train()

# ====== Save Model ======
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

merged_model = model.merge_and_unload()
merged_model.save_pretrained(merged_model_save_path)
tokenizer.save_pretrained(merged_model_save_path) 
