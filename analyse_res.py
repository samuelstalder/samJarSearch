import xml.etree.ElementTree as ET
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import csv
from preprocessing import load_colletion, load_queries



root_collection = load_colletion()
root_queries = load_queries()

query_dic = {}
for child in root_queries:
    query_dic[child[0].text] = child[1].text

documents_dic = {}
for child in root_collection:
    documents_dic[child[0].text] = child[1].text



def load_ranking(filename_rangliste):
    rangliste = []
    with open(filename_rangliste) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for row in csv_reader:
            rangliste.append([int(row[0]), row[1], int(row[2]), int(row[3])])
    return rangliste




def getTopN(queryId, rangliste, numberOfResults):
    print(f"____Query {queryId}______")
    print(query_dic[str(queryId)])
    for i in rangliste:
        for j in range(numberOfResults):
            if(i[0] == queryId and i[3] == j):
                print("Rank",j,", Doc_id" ,i[2])
                print(documents_dic[str(i[2])])
    


# Kurz: 891,245, 2405, 3777, 4767]:
# Lang: 4233,6324,8141,10893,10637
for queryId in [3777]:
    for window_size in [2,5,15]:
        print("\n__________Window Size =", window_size,"__________")
        filename_rangliste = "rankings/ranking_w"+str(window_size)+"_v200_e100.csv"
        rangliste = load_ranking(filename_rangliste)
        getTopN(queryId,rangliste, 5)

