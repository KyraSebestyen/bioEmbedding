import pandas
import gzip

token_metadata = pandas.read_csv("/disk2/ksebestyen/token_metadata1.tsv", sep = "\t", dtype = {"doi" : str})

token_metadata = token_metadata[~token_metadata.isna().any(axis = 1)]

import io
import sqlite3
import numpy

def adapt_array(arr):
    out = io.BytesIO()
    numpy.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return numpy.load(out)

sqlite3.register_adapter(numpy.ndarray, adapt_array)
sqlite3.register_converter("ARRAY", convert_array)

import sqlite3

embed_db = sqlite3.connect('/disk2/ksebestyen/embed_db_with_file.db', detect_types=sqlite3.PARSE_DECLTYPES)
cursor = embed_db.cursor()

embeddings_npy = numpy.load("/disk2/ksebestyen/embeddings1.npy")
print("embeddings loaded")

offset = 2838166 #Anzahl bereits vorhandener Datensätze +2

for idx in token_metadata.index:
    data = token_metadata.loc[idx, :].values.tolist()
    for i in range(len(data)):
        if type(data[i]) == numpy.int64:
            data[i] = int(data[i])
    text_position_start, text_position_end = [int(val.strip("()")) for val in data[4].split(", ")]
    sql_values = [data[0] + offset, data[1], data[1], data[2], data[3]] + [text_position_start, text_position_end] + data[5:]
    sql_ = '''INSERT INTO embeddings values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
    cursor.execute(sql_, tuple([*sql_values, embeddings_npy[idx, :]]))

embed_db.commit()
print("embeddings saved in database")

embed_db.close()