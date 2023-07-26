import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter


this, _ = os.path.split(__file__)
file = os.path.join(
    this,
    'data_docs',
    'MachineLearning-Lecture01.pdf'
)

loader = PyPDFLoader(file)
pages = loader.load()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
)
docs = text_splitter.split_documents(pages)
# This below shows the chunks overlapped
print(len(docs))
print(f"{docs[0].page_content[:]!r}")
print()
print(f"{docs[1].page_content[:]!r}")
print()
print(f"{docs[2].page_content[:]!r}")