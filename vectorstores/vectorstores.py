import sys
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

this, _ = os.path.split(__file__)


# # Add our third-party packages to sys.path. We've created a zip file because some of the file paths
# # are pretty long. We're also normalizing the path or we're getting import errors.

files_root = os.path.normpath(
    os.path.join(
        this,  # ./langchain-with-your-data/vectorstores
        os.pardir,  # ./langchain-with-your-data
        "data_docs",  # ./data_docs
    )
)
print(files_root)
# Load PDF
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