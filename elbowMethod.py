import pandas
import numpy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
import sqlite3

embed_db = sqlite3.connect('/disk2/ksebestyen/embed_db.db', detect_types=sqlite3.PARSE_DECLTYPES)
cursor = embed_db.cursor()

dataFromDB = cursor.execute("SELECT token_id, embedding FROM embeddings WHERE pos_univ IN ('ADJ', 'NOUN', 'PROPN') limit 10").fetchall()
dataFromDBDataFrame = pandas.DataFrame(dataFromDB)
print(dataFromDBDataFrame.head())

dataFromDBDataFrame["embedding"] = [numpy.frombuffer(entry) for entry in dataFromDBDataFrame["embedding"]]
print(dataFromDBDataFrame.head())

'''
from sklearn import preprocessing  # to normalise existing X
X_Norm = preprocessing.normalize(X)

km2 = cluster.KMeans(n_clusters=5,init='random').fit(X_Norm)


distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(dataFromDBDataFrame)
    distortions.append(kmeanModel.inertia_)
'''