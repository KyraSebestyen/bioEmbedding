import pandas
import numpy
from tqdm.notebook import tqdm
import os
import gzip
import json
import seaborn
import matplotlib.pyplot
import flair
from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings
import torch
import spacy

# %%
lossData = []

with open('/disk2/ksebestyen/checkpoint-17000/trainer_state.json') as file:
  checkpointData = json.load(file)
checkpointData = pandas.DataFrame(checkpointData["log_history"])
lossData.append(checkpointData)

lossData = pandas.concat(lossData)

# %%
matplotlib.pyplot.figure(figsize = (12, 6))
seaborn.set_style("whitegrid")

seaborn.lineplot(x = "step", y = "loss", data = lossData, label = "training loss")
seaborn.lineplot(x = "step", y = "eval_loss", data = lossData, label = "validation loss")

# %%
selectedBert = "/disk2/ksebestyen/checkpoint-17000"

# %%
corpus = pandas.read_csv('/disk2/ksebestyen/Okkurrenzen_Auszug.csv', sep = ';', quoting = 3) # 3 means QUOTE_NONE

# %%
nlp = spacy.load('en_core_web_sm')

# %%
nlp.disable_pipes(["parser", "ner"])
# nlp.add_pipe(nlp.create_pipe("sentencizer") , after = "tagger") # not needed because corpus is provided sentence wise

# %%
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


# %%
def get_batches(lst, batch_size):
  for i in range(0, len(lst), batch_size): 
    yield lst[i : i + batch_size] # gibt stückweise Elemente der Liste (1 Batch pro Iteration)

# %%
def get_sentences(doc):
  current_sentence = 0
  sentence = []
  for t in doc:
    if t["sentence_id"] == current_sentence:
      sentence.append(t["text"])
    else:
      yield sentence 
      sentence = [t["text"]]
      current_sentence = t["sentence_id"]
    if len(sentence) > 0:
      yield sentence

# %%
def get_sentence_token_mapping(doc):
  sentence_token_mapping = {}

  current_sentence = 0
  token_id_in_sent = 0
  for i, t in enumerate(doc):
    if t["sentence_id"] != current_sentence:
      current_sentence = t["sentence_id"]
      token_id_in_sent = 0
    sentence_token_mapping[(current_sentence, token_id_in_sent)] = i
    token_id_in_sent += 1

  return sentence_token_mapping

# %%
from datetime import datetime

# %%
flair.device = torch.device('cuda')
bert_model = TransformerWordEmbeddings(selectedBert,
                                       subtoken_pooling = "mean",
                                       layers = "all",
                                       layer_mean = True,
                                       allow_long_sentences = False)

out_filename_embeddings = "/disk2/ksebestyen/embeddings.npy"
out_file_metadata = gzip.open("/disk2/ksebestyen/token_metadata.tsv.gz", "wt")

metadata_columns = ["token_id", "token",  "doc_token_id", "sentence_id", "text_position", "pos_penn", "pos_univ"] #lemma

out_file_metadata.write("\t".join(metadata_columns) + "\n")

embeddings = []

running_token_id = 0

for batch_indices in get_batches(corpus.index.to_list(), 5): # holt sich Indizes des Batch
    metadata = []
    print(batch_indices)
    print(datetime.now())
    
    for (occId, file), doc in zip(corpus.loc[batch_indices, ["OccId", "File"]].values, # sucht die Zeilen batch_indices und die Spalten OccId und File
                                                   nlp.pipe(corpus.loc[batch_indices, "text"].values, # sucht die Spalte text und wendet die Pipeline auf jeden Satz an
                                                   batch_size = 10)): # zip macht aus zwei Arrays ein Array von Tupeln
        
        doc = spacy_to_json(doc)
       # sent_tokens_map = get_sentence_token_mapping(doc)

        sentences = []
        for s in get_sentences(doc):
            sentences.append(Sentence(s))

        bert_model.embed(sentences)
            
        for sentenceIndex, sentence in enumerate(sentences):
            for tokenIndex, sentenceToken in enumerate(sentence):
                
               # token_id = sent_tokens_map[(sentenceIndex, tokenIndex)]
                tokenFromMap = doc[tokenIndex]
                    
               # lemma = tokenFromMap["lemma"] #lemma ist nicht im Token ???
                text = tokenFromMap["text"]
                
                # Filter out None-entries
                if type(text) != str: #or type(lemma) != str:
                    continue
                    
                # uns interessiet nur das Embedding vom Bio-Begriff
                if text != occId:
                    continue

                # Check if Flair Sentence and document tokens are aligned
                if str(text.lower().strip()) != str(sentenceToken.text.lower().strip()):
                    print("Token is not identical!", text, sentenceToken.text)
                    
                metadata.append({
                    "token_id" : running_token_id,
                    "token" : text,
                    #"lemma" : lemma,
                    "doc_token_id" : tokenIndex,
                    "sentence_id" : tokenFromMap["sentence_id"],
                    "text_position" : (tokenFromMap["start"], tokenFromMap["end"]),
                    "pos_penn" : tokenFromMap["tag"],
                    "pos_univ" : tokenFromMap["pos"],
                })
                
                embeddings.append(sentence[tokenIndex].embedding.detach().cpu().numpy())
                running_token_id += 1
    
    # Ensure embeddings can be mapped to running token index       
    assert len(embeddings) == running_token_id
    from sys import getsizeof
    print(getsizeof(embeddings))
    
    # Write metadata
    for entry in metadata:
        out_file_metadata.write("\t".join([str(entry[column]) for column in metadata_columns]) + "\n")
    
    # Save embeddings array
    numpy.save(out_filename_embeddings, numpy.vstack(embeddings), allow_pickle = False)
    
out_file_metadata.close()
        
