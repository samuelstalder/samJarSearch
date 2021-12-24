
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

print (stopwords)

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

print("data_words[0]: ", data_words[0])


# fill corpus
id2word = corpora.Dictionary(data_words)

print("id2word: ", id2word)

corpus = []
for text in data_words:
    new = id2word.doc2bow(text)
    corpus.append(new)

print (corpus[0][0:20])

word = id2word[[0][:1][0]]
print (word)


# train model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=id2word,num_topics=30,random_state=100,update_every=1,chunksize=100,passes=10,alpha="auto")


pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, mds="mmds", R=30)
vis