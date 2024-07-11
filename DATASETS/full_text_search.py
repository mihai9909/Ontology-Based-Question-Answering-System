import datasets
import requests
import urllib
import os
import time
import csv

GREEN='\033[0;32m'
PURPLE='\033[1;35m'
RED='\033[0;31m'
NC='\033[0m' # No color

API_TOKEN = os.environ['HUGGINGFACE_TOKEN']

if API_TOKEN is None:
    print(RED, 'MISSING HUGGINGFACE_TOKEN IN ENVIRONMENT VARIABLES', NC)
    exit()

output_file = 'data.csv'
dataset_name_url_safe = urllib.parse.quote_plus('Orange/lc_quad2-sparqltotext')
config = 'default'
split = 'train'
offset = '0'
length = '100'
search = 'treatment medicine doctor hospital health drug Diabetes Cancer Hypertension blood Arthritis Asthma Allergies Influenza Flu Pneumonia HIV/AIDS Stroke Heart attack Depression Fever Cough Fatigue Headache Nausea Vomiting Diarrhea Muscle Insomnia Swelling Surgery Medication Chemotherapy Radiation therapy Psychotherapy Immunotherapy Vaccination Dialysis CPR Intubation Ventilation'
search_url_safe = urllib.parse.quote_plus(search)

headers = {"Authorization": f"Bearer {API_TOKEN}"}
def uri():
    return f"https://datasets-server.huggingface.co/search?dataset={dataset_name_url_safe}&config={config}&split={split}&query={search_url_safe}&offset={offset}&length={length}"

def extract_features():
    url = f"https://datasets-server.huggingface.co/rows?dataset={dataset_name_url_safe}&config={config}&split={split}&offset=0&length=0"
    print("Requesting features form", PURPLE, url, NC)
    features = requests.get(url, headers=headers).json()['features']
    return list(map(lambda x : x['name'], features))

def query(API_URL):
    print("START GET", PURPLE, API_URL, NC)
    response = requests.get(API_URL, headers=headers)
    return response.json()

features = extract_features()

print("Start full-text search with the following text:\n", GREEN, search, NC, '\n')

rows = []
i = 0
while True:
    time.sleep(2) # don't stress their servers
    offset = str(i*100)
    i += 1
    response_data = query(uri())
    if len(response_data['rows']) == 0: # search until no rows found
        print('RETRIEVED', GREEN, '0 ROWS', NC, 'STOPPING SEARCH')
        break
    print('RETRIEVED ', GREEN, len(response_data['rows']), ' ROWS', NC, '\n')
    rows += response_data['rows']

print("Writing to ", GREEN, output_file ,NC)
with open('data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(features) #csv header
    for row in rows:
        row_list = list(map(lambda feature : row['row'][feature], features))
        writer.writerow(row_list)

print(GREEN, "SUCCESS!", NC)

