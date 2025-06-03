import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback, DataCollatorForSeq2Seq
import torch
import wandb
torch.cuda.empty_cache()

# Load the model and tokenizer
model_name = "MBZUAI-Paris/Atlas-Chat-2B"
tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side='right')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

model.to(device)
class DebugCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        print(f"Step: {state.global_step}")
# Load datasets
train = pd.read_csv('/home/jihad.rbaiti/Work_2/CPO/data/parallel_dataset/final_data_splitting/train.csv')
val = pd.read_csv('/home/jihad.rbaiti/Work_2/CPO/data/parallel_dataset/final_data_splitting/val.csv')

def process_chat_style(data):
    source_language = "Moroccan dialect"
    target_language = "English"
    data["input_text"] = data.apply(
        lambda row: f"Translate this from [{source_language}] to [{target_language}]:\n"
                    f"{source_language}: {row['darija']}\n"
                    f"{target_language}: ", axis=1
    )
    data["labels"] = data.apply(
        lambda row: f"{row['english']}", axis=1
    )
    return data

# Apply processing to train and val datasets

train = process_chat_style(train)
val = process_chat_style(val)

print('train', train['input_text'][0])
# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train)
val_dataset = Dataset.from_pandas(val)

#print(train_dataset)
# Tokenization function
def tokenize_function(row):
    # Tokenize the input and target
    inputs = tokenizer(row["input_text"], max_length=128, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(row["labels"], max_length=128, truncation=True, padding="max_length")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    label_ids = labels.input_ids

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": label_ids
    }
    return inputs

# Tokenize datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
##

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if tokenizer.chat_template is None:
    tokenizer.chat_template = SIMPLE_SFT_CHAT_TEMPLATE

##
# Remove unnecessary columns
train_dataset = train_dataset.remove_columns(["darija", "english", "input_text", "perturbation"])
val_dataset = val_dataset.remove_columns(["darija", "english", "input_text", "perturbation"])

# Set format for PyTorch tensors
train_dataset.set_format("torch", columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format("torch", columns=['input_ids', 'attention_mask', 'labels'])
# Define training arguments



training_args = TrainingArguments(
    output_dir="/home/jihad.rbaiti/lustre/pt_cloud-muhqxqc6fxo/users/jihad.rbaiti/Work_2/Atlas_chat/SFT/",
    max_steps=10000,
    eval_strategy="epoch",
    learning_rate=1e-5,
    gradient_accumulation_steps=1,
    per_device_train_batch_size=5,
    per_device_eval_batch_size=5,
    weight_decay=0.01,
    #deepspeed="./CPO_atlas/ds_config_atlas_sft.json",
    logging_steps=200,
    optim='rmsprop',
    #generation_max_length=128,
    warmup_steps=150,
    report_to='wandb',
    fp16=False,
    bf16=True if torch.cuda.is_bf16_supported() else False,
    save_strategy="steps",
    eval_steps = 500,
    #predict_with_generate=True,
    logging_first_step=True,
    save_total_limit=3,
    num_train_epochs=3,
    remove_unused_columns=False,
    run_name='ATLAS_SFT',
    #fp16=torch.cuda.is_available(),
    logging_dir="/home/jihad.rbaiti/lustre/pt_cloud-muhqxqc6fxo/users/jihad.rbaiti/Work_2/Atlas_chat/SFT/logs",
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    callbacks=[DebugCallback()]
)

# Train the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)
