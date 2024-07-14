import requests

class GraphdbQuery:
  @staticmethod
  def query(query):
    url = "http://localhost:7200/repositories/AMD"
    headers = {
      "Content-Type": 'application/sparql-query',
      "Accept": 'application/json'
    }
    response = requests.post(url, headers=headers, data=query)
    return response