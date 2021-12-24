import xml.etree.ElementTree as ET
import spacy
import json
import gensim
sent = ["classification of the waveforms of atmospherics a review with references"]

def lemmatization(root, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    texts_out = []
    for text in root:
        doc = nlp(text)
        print(doc)
        new_text = []
        for token in doc:
            print(token, token.pos_)
            if token.pos_ in allowed_postags and token.text !="am" and token.text!="pm":
                new_text.append(token.lemma_)
        final = " ".join(new_text)
        texts_out.append(final)
    return (texts_out)

print(lemmatization(sent))