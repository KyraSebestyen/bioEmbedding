import sqlite3
import math
from tqdm import tqdm
import numpy
import pandas

import io


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

# Put indexes on the columns to significantly reduce query time
# Example: CREATE INDEX lemma_year_tokenid_index ON embeddings(lemma,year,token_id);
# Example in embed_db_index.db:
embed_db = sqlite3.connect('/disk2/ksebestyen/embed_db_with_file.db', detect_types=sqlite3.PARSE_DECLTYPES)
cursor = embed_db.cursor()

embeddings = []
# Get all ids with resepect to terms/years
cursor.execute(f'SELECT LOWER(lemma), token_id,year FROM embeddings INNER JOIN metadata ON embeddings.File = metadata.workId WHERE LOWER(lemma) = "wolf"')
embeddings.extend([(term, tokenId, year) for term, tokenId, year in cursor.fetchall()])

df = pandas.DataFrame.from_dict(embeddings)
df.columns = ["term", "token_id", "year"]
df.to_csv("/disk2/ksebestyen/wolf_embeddings.csv")

embed_db.close()
