import pandas
import numpy
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

corpus = pandas.read_csv('/disk2/ksebestyen/corpusGPU1.csv', sep = ';', quoting = 3) # 3 means QUOTE_NONE

nlp = spacy.load('en_core_web_trf')

nlp.disable_pipes(["parser", "ner"])

def spacy_to_json(spacy_doc):
    doc_dict = spacy_doc.to_json()
    # print(doc_dict)
    sent_boundaries = [10000] # habe nur Sätze und brauche deshalb nicht Satzende
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
    yield lst[i : i + batch_size] # gibt stückweise Elemente der Liste (1 Batch pro Iteration)


bert_model = TransformerWordEmbeddings("/disk2/ksebestyen/checkpoint-17000",
                                       subtoken_pooling = "mean",
                                       layers = "all",
                                       layer_mean = True,
                                       allow_long_sentences = True)
bert_model = bert_model.to("cuda:1")

out_filename_embeddings = "/disk2/ksebestyen/embeddings1.npy"
out_file_metadata = gzip.open("/disk2/ksebestyen/token_metadata1.tsv.gz", "wt")

metadata_columns = ["token_id", "token",  "doc_token_id", "sentence_id", "text_position", "pos_penn", "pos_univ"] #lemma

out_file_metadata.write("\t".join(metadata_columns) + "\n")

embeddings = []
metadata = []
count = 0
for batch_indices in get_batches(corpus.index.to_list(), 25):  # holt sich Indizes des Batch
    print(count)
    count = count + 25
    sentences = []
    occIds = []
    docs = []

    # Collect the transformed Sentences into array and batch them together again for embedding.
    for (occId, file), doc in zip(corpus.loc[batch_indices, ["OccId", "File"]].values, # sucht die Zeilen batch_indices und die Spalten OccId und File
                                                   nlp.pipe(corpus.loc[batch_indices, "Sentence"].values)): # zip macht aus zwei Arrays ein Array von Tupeln
        # print(doc)
        doc = spacy_to_json(doc) #  changes the doc to a list of dicts.

        sentences.append(Sentence([x["text"] for x in doc])) # Turns the list of spacy tokens into a flair Sentence objects
        occIds.append(occId) # Remember the occId for the sentence
        docs.append(doc) # remember the spacy doc for later comparison ( OccId seems to be lemmatized?!)

    #bert_model.embed(sentences)
    with torch.no_grad():
        bert_model.embed(sentences)

    for occId, sent, doc, bi in zip(occIds, sentences, docs,batch_indices):
        found = False

        for i, (tok, doc_tok) in enumerate(zip(sent, doc)):
            # The First check is for equality the second is a fallback in case lemmatization failed. For example in some cases bird's was not split into ["bird", "'s"] by spacy
            if doc_tok["lemma"] == occId or occId in doc_tok["text"]:
                metadata.append({
                    "token_id": len(embeddings)-1, # THis id correcsponds to the entry in the final embedding array , if I understand correctly
                    "token": doc_tok["text"],
                    "doc_token_id": doc_tok["id"],
                    "sentence_id": bi, # This now references the index in the Data frame, that way it can be referenced correctly
                    "text_position": (doc_tok["start"], doc_tok["end"]),
                    "pos_penn": doc_tok["tag"],
                    "pos_univ": doc_tok["pos"],
                })
                embeddings.append(tok.embedding.detach().cpu().numpy())
                found = True
                break # No need to search the rest of the sentence



# Write metadata
for entry in metadata:
    out_file_metadata.write("\t".join([str(entry[column]) for column in metadata_columns]) + "\n")
        # Save embeddings array
numpy.save(out_filename_embeddings, numpy.vstack(embeddings), allow_pickle=False)
out_file_metadata.close()
