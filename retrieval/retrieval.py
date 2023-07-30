# Vectorstore retrieval
import sys
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import chromadb
from langchain.embeddings.openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings()


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
# len(splits[0].page_content)
# Out[2]: 1499
splits = text_splitter.split_documents(docs)
print(len(splits))


persist_directory = os.path.join(os.path.dirname(__file__), 'chroma')
client = chromadb.PersistentClient(path=persist_directory)
# Create the Vectorstore
vectordb = Chroma.from_documents(
    client=client,
    documents=splits,
    embedding=embedding,
)
print(vectordb._collection.count())

# Similarity Search
question = "is there an email i can ask for help"
docs = vectordb.similarity_search(question, k=3)
print(len(docs))

# Similarity Search
texts = [
    """The Amanita phalloides has a large and imposing epigeous (aboveground) fruiting body (basidiocarp).""",
    """A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.""",
    """A. phalloides, a.k.a Death Cap, is one of the most poisonous of all known mushrooms.""",
]
# Let's create a small database
smalldb = Chroma.from_texts(texts, embedding=embedding)

question = "Tell me about all-white mushrooms with large fruiting bodies"
# Only retrieve the two more relevant documents
smalldb.similarity_search(question, k=2)
# Here we can see that there is no mention of the fact that is poisonous
# Out[2]:
# [Document(page_content='A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.', metadata={}),
#  Document(page_content='The Amanita phalloides has a large and imposing epigeous (aboveground) fruiting body (basidiocarp).', metadata={})]

#let's try with MMR
smalldb.max_marginal_relevance_search(question,k=2, fetch_k=3)
# Here we can see that the info that is poisonous is returned among the documents
# Out[1]:
# [Document(page_content='A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.', metadata={}),
#  Document(page_content='A. phalloides, a.k.a Death Cap, is one of the most poisonous of all known mushrooms.', metadata={})]