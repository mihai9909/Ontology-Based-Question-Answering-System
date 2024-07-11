from datasets import load_dataset
import re

dataset = load_dataset("mihai9909/Age-related-Macular-Degeneration-NL2SPARQL")

def write_to_file(file_name, data, key):
    with open(file_name, 'w') as f:
        for entry in data:
            f.write(re.sub(r'\s+', ' ',entry[key].replace("\t", ' ').replace("\n", ' ')) + '\n')

write_to_file('/app/DATASETS/train/data.lang', dataset['train'], 'NL_Query')
write_to_file('/app/DATASETS/train/data.sparql', dataset['train'], 'SPARQL')

write_to_file('/app/DATASETS/test/data.lang', dataset['test'], 'NL_Query')
write_to_file('/app/DATASETS/test/data.sparql', dataset['test'], 'SPARQL')

