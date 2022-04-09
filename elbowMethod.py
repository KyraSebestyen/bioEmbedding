import pandas
import numpy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn import preprocessing  # to normalise existing X
import sqlite3

embed_db = sqlite3.connect('/disk2/ksebestyen/embed_db.db', detect_types=sqlite3.PARSE_DECLTYPES)
cursor = embed_db.cursor()

dataFromDB = cursor.execute("SELECT token_id, embedding FROM embeddings WHERE pos_univ IN ('ADJ', 'NOUN', 'PROPN') limit 100000").fetchall()
dataFromDBDataFrame = pandas.DataFrame(dataFromDB)
dataFromDBDataFrame.columns=["token_id", "embedding"]

# print(dataFromDBDataFrame.head())

dataFromDBDataFrame["embedding"] = [numpy.frombuffer(entry) for entry in dataFromDBDataFrame["embedding"]]
# print(dataFromDBDataFrame.head())

embeddings = dataFromDBDataFrame["embedding"].tolist()
# print(embeddings)

embeddingsNumpy = numpy.array(embeddings)
# print(embeddingsNumpy)

embeddingsNormalized = preprocessing.normalize(embeddingsNumpy, axis=0)
# print(embeddingsNormalized)


distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(embeddingsNormalized)
    distortions.append(kmeanModel.inertia_)

print(distortions)
