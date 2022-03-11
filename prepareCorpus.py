import pandas
import numpy

corpus = pandas.read_csv('/disk2/ksebestyen/validCorpus.csv', sep=';', quoting=3)
corpus['Sentence'] = corpus['Sentence'].astype(str)
corpus = corpus[corpus.Sentence.map(len) < 620]

corpusList = numpy.array_split(corpus, 2)
corpusList[0].to_csv('/disk2/ksebestyen/corpusGPU0.csv', sep=';', quoting=3)
corpusList[1].to_csv('/disk2/ksebestyen/corpusGPU1.csv', sep=';', quoting=3)
