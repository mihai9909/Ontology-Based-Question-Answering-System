import requests
import jellyfish
import csv
import json
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu

dataset = load_dataset("mihai9909/Age-related-Macular-Degeneration-NL2SPARQL")

def evaluate(sparql):
    url = 'http://localhost:7200/repositories/AMD'
    headers = {
      "Content-Type": 'application/sparql-query',
      "Accept": 'application/json'
    }
    response = requests.post(url, headers=headers, data=sparql)
    if 'boolean' in json.loads(response.text):
      return [json.loads(response.text)['boolean']]
    return [row['s']['value'] for row in json.loads(response.text)['results']['bindings']]

def generate_sparql(prompt):
    url = 'http://localhost:5000/rag/ask'
    headers = {
      "Content-Type": 'application/json',
      "Accept": 'application/json'
    }
    prompt = json.dumps({"query": prompt})
    response = requests.post(url, headers=headers, data=prompt)
    return json.loads(response.text)

def jaccard(expected_result, model_result):
    expected_result = set(expected_result)
    model_result = set(model_result)
    intersection = expected_result.intersection(model_result)
    union = expected_result.union(model_result)
    if len(union) == 0:
        return 1.0
    return float(len(intersection))/float(len(union))

def dice_similarity(expected_result, model_result):
    expected_result = set(expected_result)
    model_result = set(model_result)
    intersection = expected_result.intersection(model_result)
    if len(expected_result) + len(model_result) == 0:
        return 1.0
    return 2 * float(len(intersection)) / (len(expected_result) + len(model_result))

test_queries = [query['SPARQL']  for query in dataset['test']]
test_questions = [query['NL_Query']  for query in dataset['test']]

similarity_sum = 0
for i in range(len(test_queries)):
    system_result = generate_sparql(test_questions[i])['SPARQL']
    sentence_bleu_score = sentence_bleu([test_queries[i]], system_result)
    print(f"BLEU similarity (row {i}): ", sentence_bleu_score)
    similarity_sum += sentence_bleu_score

print("Benchmark result (average BLEU similarity): ", similarity_sum / len(test_queries))

    
