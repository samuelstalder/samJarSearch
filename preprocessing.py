import xml.etree.ElementTree as ET
import spacy
import json
import gensim

def load_as_json(filename):
    with open(filename) as json_file:
        return json.load(json_file)

def store_as_json(filename, json_file):
    with open(filename, 'w') as outfile:
        json.dump(json_file, outfile)

def load_colletion():
    filename_collection = "ie1_collection.trec"
    tree = ET.parse(filename_collection)
    return tree.getroot()

def load_queries():
    filename_queries = "ie1_queries.trec"
    tree = ET.parse(filename_queries)
    return tree.getroot()

def lemmatization(root, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    texts_out = []
    for text in root:
        doc = nlp(text[1].text)
        new_text = []
        for token in doc:
            if token.pos_ in allowed_postags and token.text !="am" and token.text!="pm":
                new_text.append(token.lemma_)
        final = " ".join(new_text)
        texts_out.append(final)
    return (texts_out)

def gen_words(texts):
    final = []
    for text in texts:
        new = gensim.utils.simple_preprocess(text, deacc=True)
        final.append(new)
    return (final)

if __name__ == "__main__":

    root_collection = load_colletion()
    root_queries = load_queries()

    query_ids = []
    for query in root_queries:
        query_ids.append(query[0].text)

    doc_ids = []
    for doc in root_collection:
        doc_ids.append(doc[0].text)

    store_as_json("processed_data_json/doc_ids.json", doc_ids)
    store_as_json("processed_data_json/query_ids.json", query_ids)
    store_as_json("processed_data_json/collection_data_words.json", gen_words(lemmatization(root_collection)))
    store_as_json("processed_data_json/query_data_words.json", gen_words(lemmatization(root_queries)))

# Loading the data from json:
"""
from preprocessing import load_as_json
colletion_data_words = load_as_json("processed_data_json/collection_data_words.json")
query_data_words = load_as_json("processed_data_json/query_data_words.json")
doc_ids = load_as_json("processed_data_json/doc_ids.json")
query_ids = load_as_json("processed_data_json/query_ids.json")
"""
