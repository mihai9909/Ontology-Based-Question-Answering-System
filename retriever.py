from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import os

server_fifo_path = '/tmp/server_retriever_fifo'
llm_fifo_path = '/tmp/retriever_llm_fifo'

db = FAISS.load_local("AMD_INDEX",
                      HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'),
                      allow_dangerous_deserialization=True)

if not os.path.exists(server_fifo_path):
    os.mkfifo(server_fifo_path)

# Create the FIFO if it doesn't exist
if not os.path.exists(llm_fifo_path):
    os.mkfifo(llm_fifo_path)

with open(server_fifo_path, 'r') as server_fifo:
    while True:
        # Read the query from the server
        query = server_fifo.read().strip()
        if (not query) or query == '':
            continue
        print('Received Query: ', query)

        # Search for similar documents
        docs = db.similarity_search(query)

        context = ''.join(doc.page_content for doc in docs[:3])
        instruction = f'<s>[INST] Given this ontology:\n {context} \n\n Write a SPARQL query that answers this question:\n {query} [/INST]'

        # Pass the instruction to the LLM
        with open(llm_fifo_path, 'w') as fifo:
            fifo.write(instruction)
