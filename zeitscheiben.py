import numpy
import pandas
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as pyplot


def convertEmbeddingToNumpy(embedding):
    embeddingList = [float(value) for value in embedding[1:-1].split(", ")]
    return numpy.array(embeddingList)


globalVectorDF = pandas.read_csv("globalerVektorMinPooling.csv", sep=",", usecols=["embedding"])
globalVectorString = globalVectorDF["embedding"][0]
globalVector = convertEmbeddingToNumpy(globalVectorString)
#print(globalVector)

globalVectorDF = pandas.read_csv("globalerVektorMinPooling.csv", sep=",", usecols=["embedding"])

vectorsDF = pandas.read_csv("embedding_i10.csv", sep=",", usecols=["embedding", "year"])
vectorsDF["year"] = [decade[1:5] for decade in vectorsDF["year"]]
intervalVectors = [convertEmbeddingToNumpy(embedding) for embedding in vectorsDF["embedding"]]
#print(intervalVectors[0])

cosineMatrix = sklearn.metrics.pairwise.cosine_similarity([globalVector], intervalVectors)
#print(cosineMatrix)

pyplot.rcParams["figure.figsize"] = [7.50, 5.50]
pyplot.rcParams["figure.autolayout"] = True

pyplot.title("Cosine matrix")
pyplot.plot(vectorsDF["year"], cosineMatrix[0])

pyplot.show()

#HÄUFIGKEIT NORMALISIEREN UND MIT AUF DIE ANDERE ACHSE SETZEN
# 'STATT ÜBER ZEILEN AGGREGIEREN; GEHT AUCH ZEILE FÜR ZEILE: DAFÜR BEI agg(0:i+1) stattdessen agg(i:i+1)'
#metadaten als data frame und dann join
