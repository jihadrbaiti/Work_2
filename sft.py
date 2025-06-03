import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("MBZUAI-Paris/Atlas-Chat-2B")
model = AutoModelForCausalLM.from_pretrained("MBZUAI-Paris/Atlas-Chat-2B")

train_df = pd.read_csv('/home/jihad.rbaiti/Work_2/CPO/data/parallel_dataset/final_data_splitting/train.csv')
val_df = pd.read_csv('/home/jihad.rbaiti/Work_2/CPO/data/parallel_dataset/final_data_splitting/val.csv')

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Preprocessing function with chat template
def preprocess_function(examples):
    inputs = []
    targets = []
    for darija, english in zip(examples["darija"], examples["english"]):
        input_text = f"System: You are a helpful AI assistant that translates Moroccan dialect to English.\n\nUser: {darija}\n\nAssistant:"
        inputs.append(input_text)
        targets.append(english)  # Keep targets as plain English

    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    model_target = tokenizer.batch_encode_plus(targets, max_length=512, truncation=True, padding="max_length")

    model_inputs["labels"] = model_target["input_ids"]
    return model_inputs

# Apply preprocessing
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="/home/jihad.rbaiti/lustre/aim_neural-7he0p8agska/users/jihad.rbaiti/Work2_vf/Atlas_chat/SFT",
    max_steps=10000,
    eval_strategy="epoch",
    per_device_train_batch_size=4,  # Adjust based on your GPU
    per_device_eval_batch_size=4,   # Adjust based on your GPU
    gradient_accumulation_steps=4,  # Helps with limited memory
    num_train_epochs=3,             # Adjust as needed
    save_strategy="epoch",
    evaluation_strategy="epoch",
    optim='rmsprop',
    logging_steps=200,
    learning_rate=5e-5,            # Adjust as needed
    weight_decay=0.01,             # Adjust as needed
    warmup_steps=150,             # Adjust as needed
    run_name='ATLAS_SFT',
    fp16=True,                   # Use mixed precision if your GPU supports it
    push_to_hub=False,            # Set to True if you want to push to Hugging Face Hub
    report_to="wandb",             # Uncomment to use Weights & Biases
    logging_dir="/home/jihad.rbaiti/lustre/aim_neural-7he0p8agska/users/jihad.rbaiti/Work2_vf/Atlas_chat/SFT/logs",
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)

print("Fine-tuning complete. Model saved to ./atlas_finetuned")