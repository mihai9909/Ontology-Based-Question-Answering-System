from langchain_core.documents import Document
import torch
import re

from transformers import (AutoTokenizer,
                          AutoConfig,
                          AutoModelForSequenceClassification,
                          AutoModelForCausalLM,
                          TrainingArguments,
                          BitsAndBytesConfig,
                          pipeline,
                          logging)

from peft import (PeftModel,
                  PeftConfig,
                  get_peft_model,
                  LoraConfig)

import os

def extract_sparql(text):
    pattern = re.compile(r'```(.*?)```', re.DOTALL)
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return None

model_checkpoint = "WizardLLM"

# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
                model_checkpoint,
                return_dict=True,
                torch_dtype=torch.float16).to('cuda')

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

pipe = pipeline(task="text-generation", model=base_model, tokenizer=tokenizer, max_new_tokens=500)

read_fifo_path = '/tmp/retriever_llm_fifo'
response_fifo_path = '/tmp/llm_server_fifo'

# Create the FIFO if it doesn't exist
if not os.path.exists(read_fifo_path):
    os.mkfifo(read_fifo_path)

if not os.path.exists(response_fifo_path):
    os.mkfifo(response_fifo_path)

# Read the instruction from the FIFO

with open(read_fifo_path, 'r') as fifo:
    while True:
        instruction = fifo.read().strip()
        if (not instruction) or instruction == '':
            continue
        result = pipe(instruction)
        response = extract_sparql(result[0]['generated_text'])
        open(response_fifo_path, 'w').write(response)