from dataclasses import dataclass, field 
from accelerate import PartialState, infer_auto_device_map, init_empty_weights 
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser 
from trl import CPOConfig, CPOTrainer, ModelConfig, get_peft_config 
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE,SIMPLE_SFT_CHAT_TEMPLATE
import torch
import pandas as pd
import gc
import wandb
from torch.optim import RMSprop
from peft import LoraConfig, get_peft_model
#from peft import print_trainable_parameters 


peft_config = LoraConfig(
    r=8,  # Low-rank dimension (try 8 or 16)
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",  # since you're using a decoder-only model
)

base_model = AutoModelForCausalLM.from_pretrained("MBZUAI-Paris/Atlas-Chat-2B", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("MBZUAI-Paris/Atlas-Chat-2B")
model = get_peft_model(base_model, peft_config)
wandb.login(key='86570a60523435fb4d496c0e63e8ae11c308bae2')
print('This script is the training script of CPO using ATLAS-CHAT')
model.to(dtype=torch.bfloat16)
cpo_config = CPOConfig(
    output_dir='/home/jihad.rbaiti/lustre/aim_neural-7he0p8agska/users/jihad.rbaiti/Work2_vf/Atlas_chat_2/CPO/',
    max_steps=10000,
    eval_strategy="epoch",
    learning_rate=1e-5,
    gradient_accumulation_steps=1,
    per_device_train_batch_size=5,# default is 10
    per_device_eval_batch_size=5, # default is 10
    logging_steps=100,
    weight_decay=0.01,
    optim='rmsprop',
    warmup_steps=150,
    report_to='wandb',
    bf16=True,
    logging_first_step=True,
    beta=0.3,
    run_name='ATLAS_CPO_2',
    cpo_alpha=0.8, #previous 0.8
    eval_steps = 200,
    generate_during_eval =True, 
    max_completion_length=256,
    remove_unused_columns=False,
    max_length=256,
    max_prompt_length=256,
    logging_dir="/home/jihad.rbaiti/lustre/aim_neural-7he0p8agska/users/jihad.rbaiti/Work2_vf/Atlas_chat_2/CPO/logs")

train = pd.read_csv('/home/jihad.rbaiti/Work_2/CPO/data/parallel_dataset/final_data_splitting/train.csv')
val = pd.read_csv('/home/jihad.rbaiti/Work_2/CPO/data/parallel_dataset/final_data_splitting/val.csv')
train.rename(columns={'darija': 'prompt','english': 'chosen', 'perturbation': 'rejected'}, inplace=True)
val.rename(columns={'darija': 'prompt','english': 'chosen', 'perturbation': 'rejected'}, inplace=True)
train = Dataset.from_pandas(train)
val = Dataset.from_pandas(val)
def process(row):
    try:
        # Generate the translation prompt
        source_language = "Moroccan dialect"
        target_language = "English"
        #system = f'You are a native speaker of both [{source_language}] and [{target_language}]. You are an expert post editor of translations from [{source_language}] into [{target_language}] and a helpful assistant dedicated to improving translation quality. You will be provided with a source sentence in [{source_language}] and its translation in [{target_language}]. Your task is to carefully analyze the provided source sentence and translation, and suggest improvements to the translation. Note that you only need to generate a refined translation in [{target_language}] and do not generate anything else.'
        prompt = f"Translate this from [{source_language}] to [{target_language}]:\n{source_language}: {row['prompt']}\n{target_language}: "
        row["chosen"] = [{"role": "user", "content": prompt}, {"role": "assistant", "content": row['chosen']}]
        row["rejected"] = [{"role": "user", "content": prompt}, {"role": "assistant", "content": row['rejected']}]
        
        return row
    except Exception as e:
        print(f"Error processing row: {e}")
        return None
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
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

ds_train = train.map(process, batch_size=200, num_proc=4)
ds_val = val.map(process, batch_size=200, num_proc=4)

'''
#
# Set pad_token to eos_token if it's not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if tokenizer.chat_template is None:
    tokenizer.chat_template = SIMPLE_SFT_CHAT_TEMPLATE
### Start not correct 
'''

#tokenizer.chat_template = chat_template

trainer = CPOTrainer(
    model=model,
    args=cpo_config,
    train_dataset=ds_train,
    eval_dataset=ds_val,
    processing_class=tokenizer,
    peft_config=peft_config,
)
trainer.train()
trainer.save_model(cpo_config.output_dir)
tokenizer.save_pretrained(cpo_config.output_dir)