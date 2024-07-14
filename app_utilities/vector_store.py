from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

hf_embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

class VectorStore:
  def __init__(self):
      self.text_splitter = CharacterTextSplitter(
                              separator="\n\n\n",
                              chunk_size=1000,
                              chunk_overlap=0,
                              length_function=len,
                              is_separator_regex=False)

  def create(self, path):
      with open(f"./ontologies/{path}.rdf") as f:
          ontology = f.read()
      documents = self.text_splitter.create_documents([ontology])
      db = FAISS.from_documents(documents,
                                hf_embeddings)

      db.save_local(f"./vector_store")
      return True
