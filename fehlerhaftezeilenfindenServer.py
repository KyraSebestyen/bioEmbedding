#vorher pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.2.0/en_core_web_trf-3.2.0.tar.gz --no-deps ausführen

import pandas
import numpy
from tqdm.notebook import tqdm
import os
import gzip
import json
import seaborn
import matplotlib.pyplot
from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings
import torch

import spacy
import spacy_transformers

from datetime import datetime

nlp = spacy.load('en_core_web_trf')
nlp.disable_pipes(["parser", "ner"])

def divideCorpus (corpus):
    chunkSize = 100000
    chunkList = [corpus[i:i + chunkSize] for i in range(0, corpus.shape[0], chunkSize)]
    for i in range(len(chunkList)):
        pandas.DataFrame(chunkList[i]).to_csv('/disk2/ksebestyen/ChunkList' + str(i) + '.csv', sep=';', quoting=3)


def spacy_to_json(spacy_doc):
    doc_dict = spacy_doc.to_json()
    # print(doc_dict)
    sent_boundaries = [10000]  # habe nur Sätze und brauche deshalb nicht Satzende
    doc_dict = doc_dict["tokens"]
    current_sentence = 0
    for i, t in enumerate(spacy_doc):
        doc_dict[i]["text"] = t.text
        if spacy_doc[i].is_sent_start:
            doc_dict[i]["is_sent_start"] = True
        else:
            doc_dict[i]["is_sent_start"] = False
        if doc_dict[i]["end"] > sent_boundaries[current_sentence]:
            current_sentence += 1
        doc_dict[i]["sentence_id"] = current_sentence
    return doc_dict


def get_batches(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i: i + batch_size]  # gibt stückweise Elemente der Liste (1 Batch pro Iteration)


def categorize(corpusChunk, chunkIndex):
    valid = []
    notFound = []

    mwu = corpusChunk[
        corpusChunk['OccId'].str.match('^.*[ -].*$') == True]  # Zeile enthält Leerzeichen oder Bindestrich
    nonMWU = corpusChunk[corpusChunk['OccId'].str.match('^[^ -]+$') == True]

    # nlpPiped = list(nlp.pipe(nonMWU.loc[:, "text"].values))
    #  print(nlpPiped)

    for (index, line), doc in zip(nonMWU.iterrows(), nlp.pipe(nonMWU.loc[:, "Sentence"].values, batch_size=100)):
        jsonDoc = spacy_to_json(doc)
        found = False
        occId = line["OccId"]
        for jsonDocElement in jsonDoc:
            if jsonDocElement["lemma"] == occId or occId in jsonDocElement["text"]:
                valid.append(line)
                found = True
                break
        if not found:
            notFound.append(line)

    pandas.DataFrame(valid).to_csv('/disk2/ksebestyen/Valid' + chunkIndex + '.csv', sep=';', quoting=3)
    pandas.DataFrame(mwu).to_csv('/disk2/ksebestyen/MWU' + chunkIndex + '.csv', sep=';', quoting=3)
    pandas.DataFrame(notFound).to_csv('/disk2/ksebestyen/NotFound' + chunkIndex + '.csv', sep=';', quoting=3)


for index in range(10, 10):
    chunkIndex = str(index)
    print('Starting chunk ' + chunkIndex + ' at ' + str(datetime.now()))
    corpus = pandas.read_csv('/disk2/ksebestyen/ChunkList' + chunkIndex + '.csv', sep=';', quoting=3, dtype='str')  # 3 means QUOTE_NONE
    corpus['Sentence'] = corpus['Sentence'].astype(str)
    corpus = corpus[corpus.Sentence.map(len) < 1000]

    categorize(corpus, chunkIndex)
#divideCorpus(corpus)
