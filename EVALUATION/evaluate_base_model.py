import requests
import jellyfish
import csv 

def evaluate(sparql):
    url = 'http://localhost:7200/repositories/AMD'
    data = {
        'query': sparql,
        'infer': True,
        'sameAs': True,
        'offset': 0
        }
    headers = { 'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8' }
    response = requests.post(url, params=data, headers=headers)
    # parse CSV response
    csv_reader = csv.reader(response.text.splitlines()[1:]) # skip header
    return [row[0] for row in csv_reader]

def similarity(expected_result, model_result):
    expected_result = set(expected_result)
    model_result = set(model_result)
    intersection = expected_result.intersection(model_result)
    union = expected_result.union(model_result)
    return float(len(intersection))/float(len(union))

test_queries = open('data.sparql').read().splitlines()
model_queries = open('results_base_model.sparql').read().splitlines()

similarity_sum = 0
for i in range(len(test_queries)):
    expected_result = evaluate(test_queries[i])
    model_result = evaluate(model_queries[i])
    print("Result: ", model_result)
    print(f"Similarity (row {i}): ", similarity(expected_result, model_result))
    similarity_sum += similarity(expected_result, model_result)

print("Benchmark result (average Jaccard distance): ", similarity_sum / len(test_queries))
