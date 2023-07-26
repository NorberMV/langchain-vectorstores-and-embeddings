import sys
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import chromadb
from embeddings import embedding

# ! pip install chromadb
# # Add our third-party packages to sys.path. We've created a zip file because some of the file paths
# # are pretty long. We're also normalizing the path or we're getting import errors.

files_root = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),  # ./langchain-with-your-data/vectorstores
        os.pardir,  # ./langchain-with-your-data
        "data_docs",  # ./data_docs
    )
)

# Load PDFs
loaders = [
    # Duplicate documents on purpose - messy data
    PyPDFLoader(os.path.join(files_root, "MachineLearning-Lecture01.pdf")),
    PyPDFLoader(os.path.join(files_root, "MachineLearning-Lecture01.pdf")),
    PyPDFLoader(os.path.join(files_root, "MachineLearning-Lecture02.pdf")),
    PyPDFLoader(os.path.join(files_root, "MachineLearning-Lecture03.pdf")),
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

# Split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)

splits = text_splitter.split_documents(docs)
print(len(splits))

# Vectorstores
persist_directory = os.path.join(os.path.dirname(__file__), 'chroma')
client = chromadb.PersistentClient(path="db_metadata_v5")
vectordb = Chroma.from_documents(
    client=client,
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)
print(vectordb._collection.count())

# Similarity Search
question = "is there an email i can ask for help"
docs = vectordb.similarity_search(question,k=3)
len(docs)