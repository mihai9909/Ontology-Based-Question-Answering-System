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
model_queries = open('results.sparql').read().splitlines()

similarity_sum = 0
dice_similarity_sum = 0
bleu_score = 0
for i in range(len(test_queries)):
    expected_result = evaluate(test_queries[i])
    model_result = evaluate(model_queries[i])
    print(f"Jaccard similarity (row {i}): ", jaccard(expected_result, model_result))
    similarity_sum += jaccard(expected_result, model_result)
    print(f"Dice similarity (row {i}): ", dice_similarity(expected_result, model_result))
    dice_similarity_sum += dice_similarity(expected_result, model_result)
    print("BLEU similarity (row {i}): ", sentence_bleu([test_queries[i]], model_queries[i]))
    bleu_score += sentence_bleu([test_queries[i]], model_queries[i])

print("Benchmark result (average Dice similarity): ", dice_similarity_sum / len(test_queries))
print("Benchmark result (average Jaccard distance): ", similarity_sum / len(test_queries))
print("Benchmark result (average BLEU similarity): ", bleu_score / len(test_queries))
