
from gensim.models.doc2vec import Doc2Vec
from preprocessing import load_as_json

fname = "models/my_doc2vec_model_w5_v200_e100.model"

model = Doc2Vec.load(fname,)

filename_rangliste = "rankings/ranking_w5_v200_e100.csv"
ranking = {}
with open(filename_rangliste) as f:
    lines = f.readlines()
    for line in lines:
        line = line.split(" ")
        if line[0] not in ranking.keys():
            ranking[line[0]] = [line[2]]
        else: ranking[line[0]].append(line[2])
            

doc_ids = load_as_json("processed_data_json/doc_ids.json")
colletion_data_words = load_as_json("processed_data_json/collection_data_words.json")
query_ids = load_as_json("processed_data_json/query_ids.json")
query_data_words = load_as_json("processed_data_json/query_data_words.json")

documents = {}
for i, doc_id in enumerate(doc_ids):
    documents[doc_id] = colletion_data_words[i]

queries = {}
for i, query_id in enumerate(query_ids):
    queries[query_id] = query_data_words[i]


target_queries = [38367]
num_docs_per_query = 3
top_n_words = 3
for query_id, doc_rank_list in ranking.items():
    if int(query_id) in target_queries:
        for rank, doc_id in enumerate(doc_rank_list[:num_docs_per_query]):
            print(f"_____Query {query_id}, Document {doc_id}, rank {rank}_____")
            for query_word in queries[query_id]:                    
                similar_words = []
                for doc_word in documents[doc_id]:
                    try:
                        similarity = model.wv.similarity(query_word, doc_word)
                    except:
                        continue
                    similar_words.append({
                        "w":doc_word,
                        "sim":similarity
                    })
                similar_words.sort(key=lambda x: x["sim"], reverse=True)
                print(query_word,similar_words[:top_n_words])

