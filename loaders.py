# Just two different simple loader examples
import os
import openai
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import WebBaseLoader

openai.api_key  = os.environ['OPENAI_API_KEY']

# 1.This first example in how to load data from
# a PDF file
this, _ = os.path.split(__file__)
file = os.path.join(
    this,
    'data_docs',
    'MachineLearning-Lecture01.pdf'
)
# Load the pdf
loader = PyPDFLoader(file)
pages = loader.load()

# 2.Loading content from url's
url = (
    "https://developer.shotgridsoftware.com/tk-core/"
    "overview.html#what-is-the-toolkit-platform"
)
loader = WebBaseLoader(
)
docs = loader.load()
print(docs[0].page_content[:500])