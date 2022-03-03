import glob
import pandas
# get data file names
path = '/disk2/ksebestyen/Valid'
filenames = glob.glob(path + "/*.csv")

corpusChunks = []
for filename in filenames:
    corpusChunks.append(pandas.read_csv(filename, sep=';', quoting=3))

# Concatenate all data into one DataFrame
validCorpus = pandas.concat(corpusChunks, ignore_index=True)
validCorpus = validCorpus.drop(validCorpus.columns[[0, 1]], axis=1)
validCorpus.to_csv('/disk2/ksebestyen/validCorpus.csv', sep=';', quoting=3)
