import pandas
import gzip

# Filter out some Null-entries

with gzip.open("/disk2/ksebestyen/token_metadata0.tsv.gz") as f:
    token_metadata = pandas.read_csv(f, sep = "\t", dtype = {"doi" : str})

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

# Save to Sqlite
import sqlite3

embed_db = sqlite3.connect('/disk2/ksebestyen/embed_db.db', detect_types=sqlite3.PARSE_DECLTYPES)
cursor = embed_db.cursor()

#token_metadata[token_metadata.lemma.isin(["theory", "Theory"])]

cursor.execute('''DROP TABLE IF EXISTS embeddings''')

# Create embeddings table

sql_ = '''CREATE TABLE embeddings (
        token_id INTEGER,
        token TEXT,
        lemma TEXT,
        doc_token_id INTEGER,
        sentence_id INTEGER,
        text_position_start INTEGER,
        text_position_end INTEGER,
        pos_penn TEXT,
        pos_univ TEXT,
        embedding ARRAY
)'''
cursor.execute(sql_)

embeddings_npy = numpy.load("/disk2/ksebestyen/embeddings0.npy")

for idx in token_metadata.index:
    data = token_metadata.loc[idx, :].values.tolist()
    for i in range(len(data)):
        if type(data[i]) == numpy.int64:
            data[i] = int(data[i])
    text_position_start, text_position_end = [int(val.strip("()")) for val in data[4].split(", ")]
    sql_values = [data[0], data[1], data[1], data[2], data[3]] + [text_position_start, text_position_end] + data[5:]
    sql_ = '''INSERT INTO embeddings values (?,?,?,?,?,?,?,?,?,?)'''
    cursor.execute(sql_, tuple([*sql_values, embeddings_npy[idx, :]]))

for column in ["token_id", "token", "lemma", "sentence_id", "pos_penn", "pos_univ"]:
    sql_ = f'''CREATE INDEX {column} on embeddings ({column})'''
    cursor.execute(sql_)

embed_db.commit()

embed_db.close()