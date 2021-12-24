# Slim Shady Search
import nltk
import xml.etree.ElementTree as ET
import numpy as np


filename_collection = "ie1_collection.trec"
tree = ET.parse(filename_collection)
root_collection = tree.getroot()


filename_queries = "ie1_queries.trec"
tree = ET.parse(filename_queries)
root_queries = tree.getroot()

def getTokens(sentence_data):
  # what to do with "i'd like"
  sentence_data = str(sentence_data).replace(",", "").replace(".", "").replace("?", "").replace(";", "").replace(":", "").replace('"', "")
  sentence_data = str(sentence_data).replace("-", " ")
  sentence_data = str(sentence_data).lower()
  nltk_tokens = nltk.word_tokenize(sentence_data)
  return nltk_tokens


def getCollectionText():
  text = ""
  for child in root_collection:
    text += child[1].text
  return text

def getAllDocuments():
  documents = []
  for child in root_collection:
    documents.append(child[1].text)
  return documents


# tf (term frequency, Termhäufigkeit) Anzahl der Vorkommen des Terms k in Dokument i
# auch  genannt ff (feature frequency; Merkmalshäufigkeit)
def tf(document, term):
  tokens = getTokens(document)
  amountOfTokens = len(document)
  amountOfFoundTerms = 0
  for token in tokens:
    if(token == term): amountOfFoundTerms += 1
    term_frequency = amountOfFoundTerms / amountOfTokens
  return term_frequency

# df (document frequency, Dokumentenhäufigkeit) - in wievielen Dokumenten tritt ein Merkmal/Term auf
def df(documents, term):
  amountOfDocuments = len(documents)
  amountOfFoundDocuments = 0
  for doc in documents:
    tokens = getTokens(doc)
    for token in tokens:
      if(token == term): 
        amountOfFoundDocuments += 1
        break
  document_frequency = amountOfFoundDocuments / amountOfDocuments
  return document_frequency

# N: Anzahl der Dokumente in der Kollektion
def getAmountOfDocuments():
  return 0

# idf: Inverse Document Frequency
def idf(documents, term):
  N = len(documents)

  inverse_document_frequency = np.log((1+N) / (1+df(documents,term)))
  return inverse_document_frequency

# w: Gewichtungsfunktion
def w(documents, document, term):
  gewicht = tf(document, term) * idf(documents, term)
  return gewicht

# RSV: retrieval status value
def RSV():
  return 0


def printStats():
  all_documents = getAllDocuments()
  example_document = all_documents[4]
  example_term = "meteors"
  res_tf = tf(example_document, example_term)
  res_df = df(all_documents, example_term)

  print("Anzahl Token: ", len(getTokens(getCollectionText())))
  print("Anzahl Terms: ")
  print("Anzahl Documents: ", len(all_documents))
  print("example term: ", example_term)
  print("example document: ", example_document)
  print("tf: ", res_tf)
  print("df: ", res_df)


printStats()