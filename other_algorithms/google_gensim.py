import numpy as np
import json
import glob

#Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

#spacy
import spacy
from nltk.corpus import stopwords

#vis
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import nltk

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def load_data(file):
    with open (file, "r", encoding="utf-8") as f:
        data = json.load(f) 
    return (data)

def write_data(file, data):
    with open (file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

stopwords = stopwords.words("english")

# import data

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


def lemmatization(allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    texts_out = []
    for text in root_collection:
        doc = nlp(text[1].text)
        new_text = []
        for token in doc:
            if token.pos_ in allowed_postags:
                new_text.append(token.lemma_)
        final = " ".join(new_text)
        texts_out.append(final)
    return (texts_out)


lemmatized_texts = lemmatization()

def gen_words(texts):
    final = []
    for text in root_collection:
        new = gensim.utils.simple_preprocess(text[1].text, deacc=True)
        final.append(new)
    return (final)

data_words = gen_words(lemmatized_texts)

print(data_words[3])

# use pretrained model
import gensim.downloader as api
wv = api.load('word2vec-google-news-300')
vec_king = wv['king']
print(vec_king)


# Function returning vector reperesentation of a document
def get_embedding_w2v(doc_tokens):
    embeddings = []
    if len(doc_tokens)<1:
        return np.zeros(300)
    else:
        for tok in doc_tokens:
            if tok in wv.vocab:
                embeddings.append(wv.word_vec(tok))
            else:
                embeddings.append(np.random.rand(300))
        # mean the vectors of individual words to get the vector of the document
        return np.mean(embeddings, axis=0)

doc_avg = get_embedding_w2v(data_words[3])

print(doc_avg)
