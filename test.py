import torch
import re

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

from datasets import load_dataset


model_checkpoint = "WizardLLM"

# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_checkpoint,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map='cuda',
)
lora_weights_stage_1 = "WizardLM-LoRA-weights-SPARQL-SimpleQuestions"
lora_weights_stage_2 = "WizardLM-LoRA-weights-SPARQL-AMD"

model = PeftModel.from_pretrained(base_model, lora_weights_stage_1)
model = model.merge_and_unload(progressbar=True, safe_merge=True)

model = PeftModel.from_pretrained(model, lora_weights_stage_2)
model = model.merge_and_unload(progressbar=True, safe_merge=True)

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# input_text = "Write me a python script that creates a file and writes 10 random bytes to it"
# input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

# outputs = model.generate(input_ids, pad_token_id=tokenizer.eos_token_id, max_length=700)
# print(tokenizer.decode(outputs[0]))
'''
prompt = "What makes macular degeneration worse?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=700)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])
'''

def extract_sparql(text):
    after_inst_pattern = r'\[/INST\](.*)'
    match = re.search(after_inst_pattern, text, re.DOTALL)
    if match:
        content_after_inst = match.group(1).strip()
    else:
        content_after_inst = ""

    cleaned_content = re.sub(r'\[INST\]|\[/INST\]', '', content_after_inst)

    return cleaned_content

pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=700)

ds = load_dataset("mihai9909/Age-related-Macular-Degeneration-NL2SPARQL")
lines = ds['test']['NL_Query']
output_file = open('EVALUATION/results.sparql', 'w')

for prompt in lines:
    prompt = prompt.strip()
    result = pipe(f"[INST] {prompt} [/INST]")
    print(result[0]['generated_text'] + '\n')
    sparql_query = extract_sparql(result[0]['generated_text'])
    sparql_query = re.sub(r'\s+', ' ', sparql_query.replace("\t", ' ').replace("\n", ' '))
    output_file.write(sparql_query + '\n')

