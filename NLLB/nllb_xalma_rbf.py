from dataclasses import dataclass, field 
from accelerate import PartialState, infer_auto_device_map, init_empty_weights 
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, AutoModelForSeq2SeqLM, M2M100Config
from trl import ModelConfig, get_peft_config#,CPOConfig, CPOTrainer 
from ALMA1.utils.cpo_trainer_rbf import CPOTrainer
from ALMA1.utils.cpo_config import CPOConfig

from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE,SIMPLE_SFT_CHAT_TEMPLATE
import torch
import pandas as pd
torch.cuda.empty_cache()
import wandb
from torch.optim import RMSprop
print(torch.version.cuda)  # Should not be None
print(torch.cuda.is_available())
#device = torch.device("cuda")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print('Training script of XALMA_rbf')

model1 = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(device)#haoranxu/ALMA-13B
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
original_config = M2M100Config.from_pretrained("facebook/nllb-200-distilled-600M")
original_config.encoder_layers = 2
original_config.decoder_layers = 2

model = AutoModelForSeq2SeqLM.from_config(original_config)
#print('model_logits',model)
#wandb.init(id='bx4ip03h', resume='must', project="huggingface", name="NLLB_XALMA_rbf" )
wandb.login(key='86570a60523435fb4d496c0e63e8ae11c308bae2')
cpo_config = CPOConfig(
    output_dir='/localssd/chouaib/geo_ai/Model_NLLB/XALMA_rbf/',
    max_steps=10000,
    eval_strategy="epoch",
    max_target_length=256,
    learning_rate=1e-5,
    gradient_accumulation_steps=1,
    per_device_train_batch_size=19,# default is 10
    per_device_eval_batch_size=20, # default is 10
    logging_steps=100,
    weight_decay=0.01,#
    optim='rmsprop',
    warmup_steps=150,
    report_to='wandb',
    bf16=True,
    logging_first_step=True,
    beta=0.2,
    run_name='NLLB_XALMA_RBF',
    cpo_alpha=0.8,
    eval_steps = 200,
    generate_during_eval =True, 
    is_encoder_decoder = True, 
    max_completion_length=256,
    remove_unused_columns=False,
    max_length=200,
    max_prompt_length=256,logging_dir="/localssd/chouaib/geo_ai/Model_NLLB/XALMA_rbf/logs")
train = pd.read_csv('/home/jihad.rbaiti/Work_2/CPO/data/parallel_dataset/final_data_splitting/train.csv')#, nrows=150)
val = pd.read_csv('/home/jihad.rbaiti/Work_2/CPO/data/parallel_dataset/final_data_splitting/val.csv')#, nrows=15)
train.rename(columns={'darija': 'prompt','english': 'chosen', 'perturbation': 'rejected'}, inplace=True)
val.rename(columns={'darija': 'prompt','english': 'chosen', 'perturbation': 'rejected'}, inplace=True)
train = Dataset.from_pandas(train)
val = Dataset.from_pandas(val)

# Set tokenizer configurations
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Language codes for Moroccan Arabic and English
tokenizer.src_lang = "ary_Arab"  # Source: Moroccan Arabic
tokenizer.tgt_lang = "eng_Latn"  # Target: English

trainer = CPOTrainer(
    model=model1,
    args=cpo_config,
    train_dataset=train,
    eval_dataset=val,
    #processing_class=tokenizer,
    tokenizer=tokenizer,
)
#trainer.train(resume_from_checkpoint="/home/jihad.rbaiti/lustre/aim_neural-7he0p8agska/users/jihad.rbaiti/Work2_vf/NLLB/XALMA_rbf/checkpoint-6500")
trainer.train()
trainer.save_model(cpo_config.output_dir)
tokenizer.save_pretrained(cpo_config.output_dir)