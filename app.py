import flask
import logging
import os
from app_utilities.graphdb_uploader import GraphdbUploader
from app_utilities.graphdb_query import GraphdbQuery
from app_utilities.vector_store import VectorStore

to_retriever_fifo = '/tmp/server_retriever_fifo'
from_llm_fifo = '/tmp/llm_server_fifo'

app = flask.Flask(__name__,
                  static_folder='./static/',
                  static_url_path='/public')
app.logger.setLevel('INFO')
handler = logging.FileHandler('logs/app.log')
app.logger.addHandler(handler)

@app.route('/rag/ask', methods=['POST'])
def rag_ask():
    if 'query' not in flask.request.json:
        return flask.jsonify({"error": "No query provided"}), 400

    query = flask.request.json['query']
    if not os.path.exists(to_retriever_fifo):
        os.mkfifo(to_retriever_fifo)
    open(to_retriever_fifo, 'w').write(query)

    if not os.path.exists(from_llm_fifo):
        os.mkfifo(from_llm_fifo)
    SPARQL = open(from_llm_fifo, 'r').read()

    response = GraphdbQuery.query(SPARQL)
    if response.status_code != 200:
        return flask.jsonify({"error": "Error querying the ontology", "SPARQL": SPARQL, 'text': response.text }), 200
    
    print(SPARQL)
    return { 'text': response.text, 'SPARQL': SPARQL }, 200

UPLOAD_FOLDER = './ontologies'
if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

@app.route('/query')
def query():
    if 'query' not in flask.request.json:
        return flask.jsonify({"error": "No query provided"}), 400

    query = flask.request.json['query']
    return GraphdbQuery.query(query).text, 200

from_trained_model_fifo = '/tmp/trained-model_server_fifo'
to_trained_model_fifo = '/tmp/server_trained-model_fifo'

@app.route('/ask', methods=['POST'])
def ask():
    if 'query' not in flask.request.json:
            return flask.jsonify({"error": "No query provided"}), 400

    query = flask.request.json['query']
    
    if not os.path.exists(to_trained_model_fifo):
        os.mkfifo(to_trained_model_fifo)
    
    open(to_trained_model_fifo, 'w').write(query)

    if not os.path.exists(from_trained_model_fifo):
        os.mkfifo(from_trained_model_fifo)
    
    SPARQL = open(from_trained_model_fifo, 'r').read()

    response = GraphdbQuery.query(SPARQL)
    if response.status_code != 200:
        return flask.jsonify({"error": "Error querying the ontology", "SPARQL": SPARQL, 'text': response.text}), 200
    
    app.logger.info(SPARQL)
    return response.text, 200

@app.route('/ontologies/upload', methods=['POST'])
def upload_rdf():
        if 'rdf_file' not in flask.request.files:
            return flask.jsonify({"error": "No file part"}), 400
                    
        file = flask.request.files['rdf_file']
                        
        if file.filename == '':
            return flask.jsonify({"error": "No selected file"}), 400

        if file and file.filename.endswith('.rdf'):
            filename = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filename)
            file_basename = file.filename.split('.')[0]
            response = GraphdbUploader.upload(file_basename)
            print(response.text)
            if not VectorStore().create(file_basename):
                return flask.jsonify({"error": "Error creating vector store"}), 500
            else:
                return flask.jsonify({"message": "File successfully uploaded and indexed"}), 200
        else:
            return flask.jsonify({"error": "Invalid file type, only .rdf files are allowed"}), 400
