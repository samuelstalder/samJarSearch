
import pandas as pd
import numpy as np
import dask.dataframe as dd
import nltk
import xml.etree.ElementTree as ET
from gensim.models import Word2Vec
from gensim.test.utils import common_texts


filename_collection = "ie1_collection.trec"
tree = ET.parse(filename_collection)
root_collection = tree.getroot()


filename_queries = "ie1_queries.trec"
tree = ET.parse(filename_queries)
root_queries = tree.getroot()

text = ""
quer_col = {}

def getTokens(sentence_data):
  # what to do with "i'd like"
  sentence_data = str(sentence_data).replace(",", "").replace(".", "").replace("?", "").replace(";", "").replace(":", "").replace('"', "")
  sentence_data = str(sentence_data).replace("-", " ")
  sentence_data = str(sentence_data).lower()
  nltk_tokens = nltk.word_tokenize(sentence_data)
  return nltk_tokens


# put into a corpus

df = pd.DataFrame(columns=['recordId', 'text'])

for child in root_collection:
    quer_col[child[0].text] = child[1].text
    text += child[1].text

tokens = getTokens(text)

#print(tokens)

# Create CBOW model
w2v_model = Word2Vec(tokens, vector_size=300, min_count=2, window=5, sg=1, workers=4)

print(w2v_model)
