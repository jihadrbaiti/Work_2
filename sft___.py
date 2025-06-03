# Set environment variables for detailed logging
import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from accelerate import Accelerator
import wandb
import time

# PEFT imports
from peft import LoraConfig, get_peft_model

# Initialize Weights & Biases
wandb.login(key='86570a60523435fb4d496c0e63e8ae11c308bae2')
wandb.init(project="huggingface", name="ATLAS_SFT", config={"logging_first_step": True}, mode="online")
torch.cuda.empty_cache()

# Open loss log file
log_file_path = "loss_log_atlas_sft.txt"
loss_log_file = open(log_file_path, "w+")

# Dataset definition
class TranslationDataset(Dataset):
    def __init__(self, tokenizer, data, max_length=512):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source_language = "Moroccan dialect"
        target_language = "English"
        row = self.data.iloc[idx]
        source_text = f"Translate this from [{source_language}] to [{target_language}]:\n" \
                      f"{source_language}: {row['darija']}\n" \
                      f"{source_language}: "
        target_text = row['english']

        source_encoding = self.tokenizer(source_text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        target_encoding = self.tokenizer(target_text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")

        return {
            'input_ids': source_encoding['input_ids'].squeeze(0),
            'attention_mask': source_encoding['attention_mask'].squeeze(0),
            'labels': target_encoding['input_ids'].squeeze(0)
        }

# Initialize model and tokenizer with LoRA PEFT
model_id = "MBZUAI-Paris/Atlas-Chat-2B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, peft_config)
model.to(dtype=torch.bfloat16)

print(model.print_trainable_parameters())

# Accelerator setup
accelerator = Accelerator()

# Load data
train = pd.read_csv('/home/jihad.rbaiti/Work_2/CPO/data/parallel_dataset/final_data_splitting/train.csv')
val = pd.read_csv('/home/jihad.rbaiti/Work_2/CPO/data/parallel_dataset/final_data_splitting/val.csv')

# Prepare datasets and loaders
dataset_train = TranslationDataset(tokenizer, train)
dataset_val = TranslationDataset(tokenizer, val)
train_loader = DataLoader(dataset_train, batch_size=5, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=5, shuffle=False)

# Optimizer
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-5)

# Use accelerator.prepare correctly
model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
model.train()

wandb.watch(model, log="all")

# Validation function
def validate(model, val_loader, step):
    model.eval()
    total_loss = 0
    start_time = time.time()
    with torch.no_grad():
        for batch in val_loader:
            inputs = {
                "input_ids": batch['input_ids'],
                "attention_mask": batch['attention_mask'],
            }
            labels = batch['labels'].contiguous()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
    val_loss = total_loss / len(val_loader)
    validation_runtime = time.time() - start_time
    wandb.log({"eval/loss": val_loss, "eval/step": step, "eval/runtime": validation_runtime})
    return val_loss

# Initial validation
initial_val_loss = validate(model, val_loader, step=0)
wandb.log({"step": 0, "validation_loss": initial_val_loss})
if accelerator.is_main_process:
    loss_log_file.write(f"Step 0, Validation Loss: {initial_val_loss:.6f}\n")
    loss_log_file.flush()

# Training loop
total_steps = 10000
validate_every = 200
current_step = 0

while current_step < total_steps:
    for batch in train_loader:
        if current_step >= total_steps:
            break

        start_time = time.time()
        inputs = {
            "input_ids": batch['input_ids'],
            "attention_mask": batch['attention_mask'],
        }
        labels = batch['labels'].contiguous()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        step_runtime = time.time() - start_time

        if accelerator.is_main_process:
            wandb.log({"train/loss": loss.item(), "train/step": current_step, "train/runtime": step_runtime})
            print(f"Step {current_step+1}, Loss: {loss.item()}")
            loss_log_file.write(f"Step {current_step}, Train Loss: {loss.item():.6f}\n")
            loss_log_file.flush()

        if current_step % validate_every == 0:
            val_loss = validate(model, val_loader, step=current_step)
            if accelerator.is_main_process:
                print(f"Validation Loss after {current_step+1} steps: {val_loss}")
                loss_log_file.write(f"Step {current_step}, Validation Loss: {val_loss:.6f}\n")
                loss_log_file.flush()

            if current_step != total_steps:
                checkpoint_path = f"/home/jihad.rbaiti/lustre/pt_cloud-muhqxqc6fxo/users/jihad.rbaiti/Work_2/Atlas_chat/SFT/model_checkpoint_step_{current_step}/"
                os.makedirs(checkpoint_path, exist_ok=True)
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(checkpoint_path, save_adapter=True)
                tokenizer.save_pretrained(checkpoint_path)
                torch.save(optimizer.state_dict(), checkpoint_path + 'optimizer_state.pt')

        current_step += 1

# Save final model adapter and tokenizer
model_path = f"/home/jihad.rbaiti/lustre/aim_neural-7he0p8agska/users/jihad.rbaiti/Work2_vf/Atlas_chat/SFT/model_checkpoint_step_{total_steps}/"
os.makedirs(model_path, exist_ok=True)
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(model_path, save_adapter=True)
tokenizer.save_pretrained(model_path)
torch.save(optimizer.state_dict(), model_path + 'optimizer_state.pt')

# Close resources
loss_log_file.close()
wandb.finish()
