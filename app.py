import flask
import logging
import os
from app_utilities.graphdb_uploader import GraphdbUploader

to_retriever_fifo = '/tmp/server_retriever_fifo'
from_llm_fifo = '/tmp/llm_server_fifo'

app = flask.Flask(__name__)
app.logger.setLevel('INFO')
handler = logging.FileHandler('app.log')
app.logger.addHandler(handler)

@app.route('/query', methods=['POST'])
def query():
    query = flask.request.json['query']
    if not os.path.exists(to_retriever_fifo):
        os.mkfifo(to_retriever_fifo)
    # app.logger.info(f'Query: {query}')
    open(to_retriever_fifo, 'w').write(query)

    if not os.path.exists(from_llm_fifo):
        os.mkfifo(from_llm_fifo)
    SPARQL = open(from_llm_fifo, 'r').read()
    return flask.jsonify({'sparql': SPARQL}), 200

UPLOAD_FOLDER = './ontologies'
if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

@app.route('ontologies/upload', methods=['POST'])
def upload_rdf():
        if 'rdf_file' not in request.files:
            return jsonify({"error": "No file part"}), 400
                    
        file = request.files['rdf_file']
                        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
                                        
        if file and file.filename.endswith('.rdf'):
            filename = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filename)
            GraphdbUploader
            return jsonify({"message": "File successfully uploaded"}), 200
        else:
            return jsonify({"error": "Invalid file type, only .rdf files are allowed"}), 400
