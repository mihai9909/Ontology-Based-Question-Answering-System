from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import os

server_fifo_path = '/tmp/server_retriever_fifo'
llm_fifo_path = '/tmp/retriever_llm_fifo'

if not os.path.exists(server_fifo_path):
    os.mkfifo(server_fifo_path)

if not os.path.exists(llm_fifo_path):
    os.mkfifo(llm_fifo_path)

db = FAISS.load_local("vector_store",
                      HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'),
                      allow_dangerous_deserialization=True)

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
        instruction = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Given this ontology:
{context}
Write a SPARQL query that answers the following question. Respond only with the SPARQL query and nothing else.
{query}

### Response:"""

        # Pass the instruction to the LLM
        with open(llm_fifo_path, 'w') as fifo:
            fifo.write(instruction)
