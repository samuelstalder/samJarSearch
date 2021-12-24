
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.corpus import stopwords
from numpy import vectorize
from preprocessing import  load_as_json
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


"""import os
import sys
hashseed = os.getenv('PYTHONHASHSEED')
if not hashseed:
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)"""

stopwords = stopwords.words("english")

"""
root_collection = load_colletion()
mapped_documents = []
for child in root_collection:
    mapped_documents.append(child[0].text + ": " + child[1].text)"""

colletion_data_words = load_as_json("processed_data_json/collection_data_words.json")
query_data_words = load_as_json("processed_data_json/query_data_words.json")
doc_ids = load_as_json("processed_data_json/doc_ids.json")
query_ids = load_as_json("processed_data_json/query_ids.json")

tagged_documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(colletion_data_words)]

vector_size = 200
window_size = 15
epochs = 100
model = Doc2Vec(tagged_documents, window=window_size, workers=4, vector_size=vector_size, epochs=epochs)

model.save("models/my_doc2vec_model_w"+str(window_size)+"_v"+str(vector_size)+"_e"+str(epochs)+".model")

f = open("rankings/ranking_w"+str(window_size)+"_v"+str(vector_size)+"_e"+str(epochs)+".csv", "w")

for i, query in enumerate(query_data_words):
    
    query_id = query_ids[i]
    new_vector = model.infer_vector(query)
    similar_doc = model.docvecs.most_similar([new_vector], topn=1000)

    for doc_rank, sim_doc in enumerate(similar_doc):
        doc_id = doc_ids[sim_doc[0]]
        rsv = sim_doc[1]
        team_name = "Sam.jar" 
        f.write(f"{query_id} Q0 {doc_id} {doc_rank} {rsv} {team_name}\n")