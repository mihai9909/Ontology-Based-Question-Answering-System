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

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def extract_sparql(text):
    text = text.replace("\n", ' ').replace("\t", ' ')
    after_response_pattern = r'Response:(.*)'
    after_response = re.search(after_response_pattern, text).group(1)
    return after_response

pipe = pipeline(task="text-generation", model=base_model, tokenizer=tokenizer, max_length=700)

ds = load_dataset("mihai9909/Age-related-Macular-Degeneration-NL2SPARQL")
lines = ds['test']['NL_Query']
output_file = open('DATASETS/results_base_model.sparql', 'w')

for prompt in lines:
    prompt = prompt.strip()
    result = pipe(f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
    
    ### Instruction:
    You are a SPARQL generator which generates queries for the Age-related Macular Degeneration (AMD) ontology.
    You shall answer only with the SPARQL query and nothing else.
    Translate the following sentence to SPARQL:
    {prompt}
    
    ### Response:""")
    print(result[0]['generated_text'] + '\n')
    sparql_query = extract_sparql(result[0]['generated_text'])
    print("Extracted sparql: ", sparql_query + '\n')
    sparql_query = re.sub(r'\s+', ' ', sparql_query.replace("\t", ' ').replace("\n", ' '))
    output_file.write(sparql_query + '\n')
