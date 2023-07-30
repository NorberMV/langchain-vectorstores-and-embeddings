
from langchain.embeddings.openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings()

sentence1 = "i like dogs"
sentence2 = "i like canines"
sentence3 = "the weather is ugly outside"

embedding1 = embedding.embed_query(sentence1)
embedding2 = embedding.embed_query(sentence2)
embedding3 = embedding.embed_query(sentence3)

# Comparing this we can see the difference between embeddings
# import numpy as np
# np.dot(embedding1, embedding2)
# Out[3]: 0.9631853877103519
# np.dot(embedding1, embedding3)
# Out[4]: 0.770999765129468