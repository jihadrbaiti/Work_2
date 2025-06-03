from dataclasses import dataclass, field 
from accelerate import PartialState, infer_auto_device_map, init_empty_weights 
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser 
from trl import CPOConfig, CPOTrainer, ModelConfig, get_peft_config 
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE,SIMPLE_SFT_CHAT_TEMPLATE
import torch
import pandas as pd
torch.cuda.empty_cache()
import wandb
from torch.optim import RMSprop

model = AutoModelForCausalLM.from_pretrained("/home/jihad.rbaiti/lustre/pt_cloud-muhqxqc6fxo/users/jihad.rbaiti/Work_2/Atlas_chat/SFT_inc/model_checkpoint_step_500", torch_dtype=torch.bfloat16)#haoranxu/ALMA-13B
tokenizer = AutoTokenizer.from_pretrained("/home/jihad.rbaiti/lustre/pt_cloud-muhqxqc6fxo/users/jihad.rbaiti/Work_2/Atlas_chat/SFT_inc/model_checkpoint_step_500")
wandb.login(key='86570a60523435fb4d496c0e63e8ae11c308bae2')
print('This script is the training script of SFT+CPO ATLAS-CHAT')
model.to(dtype=torch.bfloat16)
cpo_config = CPOConfig(
    output_dir='/home/jihad.rbaiti/lustre/pt_cloud-muhqxqc6fxo/users/jihad.rbaiti/Work_2/Atlas_chat/SFT+CPO/',
    max_steps=10000,
    eval_strategy="epoch",
    learning_rate=1e-5,
    gradient_accumulation_steps=1,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=10,
    logging_steps=200,
    weight_decay=0.01,#
    optim='rmsprop',
    warmup_steps=150,
    report_to='wandb',
    bf16=True,
    logging_first_step=True,
    beta=0.1,
    run_name='ATLAS_SFT+CPO',
    eval_steps = 500,
    generate_during_eval =True, 
    is_encoder_decoder = True, 
    max_completion_length=128,
    remove_unused_columns=False,
    max_length=128,
    max_prompt_length=128,logging_dir="/home/jihad.rbaiti/lustre/pt_cloud-muhqxqc6fxo/users/jihad.rbaiti/Work_2/Atlas_chat/SFT+CPO/logs")

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
        prompt = f"Translate this from [{source_language}] to [{target_language}]:\n{source_language}: {row['prompt']}\n{target_language}: "
        row["chosen"] = [{"role": "user", "content": prompt}, {"role": "assistant", "content": row['chosen']}]
        row["rejected"] = [{"role": "user", "content": prompt}, {"role": "assistant", "content": row['rejected']}]
        
        return row
    except Exception as e:
        print(f"Error processing row: {e}")
        return None

ds_train = train.map(process, num_proc=4)
ds_val = val.map(process, num_proc=4)


"""
#
# Set pad_token to eos_token if it's not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if tokenizer.chat_template is None:
    tokenizer.chat_template = SIMPLE_SFT_CHAT_TEMPLATE
### Start not correct 
"""

#tokenizer.chat_template = chat_template

trainer = CPOTrainer(
    model=model,
    args=cpo_config,
    train_dataset=ds_train,
    eval_dataset=ds_val,
    tokenizer=tokenizer,
)
trainer.train()
trainer.save_model(cpo_config.output_dir)
tokenizer.save_pretrained(cpo_config.output_dir)
