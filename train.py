import json
import re
import os

from datasets import (load_dataset,
                      Dataset)
import pandas as pd
import numpy as np
import torch

from transformers import (AutoTokenizer,
                         AutoConfig,
                         AutoModelForSequenceClassification,
                         AutoModelForCausalLM,
                         DataCollatorWithPadding,
                         TrainingArguments,
                         BitsAndBytesConfig,
                         pipeline,
                         logging)

from peft import (
    PeftModel,
    PeftConfig,
    get_peft_model,
    LoraConfig
)

from trl import SFTTrainer

from huggingface_hub import notebook_login, login, logout

BLUE='\033[1;34m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

print("Devices available: ", GREEN + str(torch.cuda.device_count()) + NC)

current_device = torch.cuda.current_device()
print("Current device: ", CYAN + str(current_device) + NC)
print("Current device name:", CYAN + str(torch.cuda.get_device_name(current_device)) + NC)

huggingface_token=os.environ.get('HUGGINGFACE_TOKEN', None)
login(huggingface_token)

model_checkpoint="WizardLLM"

base_model = AutoModelForCausalLM.from_pretrained(
    model_checkpoint,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map='cuda',
)
lora_weights = "WizardLM-LoRA-weights-SPARQL-SimpleQuestions"

print(CYAN + 'Merging ' + lora_weights + ' into ' + model_checkpoint + NC)
model = PeftModel.from_pretrained(base_model, lora_weights)
model = model.merge_and_unload(progressbar=True, safe_merge=True)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

ds = load_dataset('mihai9909/Age-related-Macular-Degeneration-NL2SPARQL')
questions = ds['train']['NL_Query']
queries = ds['train']['SPARQL']

print("Questions: ", GREEN, len(questions), NC)
print("Queries: ", GREEN, len(queries), NC)

combined_data = [f"[INST] {q} [/INST] {a}" for q, a in zip(questions, queries)]
dataset = Dataset.from_dict({"text": combined_data})

compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=0,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
)

print(GREEN + "Training..." + NC)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

if tokenizer.pad_token is None:
  tokenizer.add_special_tokens({'pad_token': '<pad>'})

trainer.train()

logging.set_verbosity(logging.CRITICAL)

new_model = 'WizardLM-LoRA-weights-SPARQL-AMD'
trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)
