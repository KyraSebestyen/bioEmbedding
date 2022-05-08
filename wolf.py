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

# ________________________________________________________________________________________________________________________
# Loading the embeddings from database
# This will take some time because of the array conversion to stream and then numpy.
from datetime import datetime

t1 = datetime.now()
a = cursor.execute(
    "SELECT token_id, embedding, year FROM embeddings INNER JOIN metadata ON embeddings.File = metadata.workId WHERE LOWER(lemma) = 'wolf'").fetchall()
e = pandas.DataFrame(a)
t2 = datetime.now()
print(f"Loading Embeddings took {(t2 - t1)}")
e.columns = ["token_id", "embedding", "year"]
# ------------------------------------------------------------------------------------------------------------------------

# The max index seems to match:
print(e.shape, cursor.execute("SELECT MAX(token_id) year FROM embeddings").fetchall())
print(e.shape, cursor.execute("SELECT MIN(token_id) year FROM embeddings").fetchall())
embeddings = []
# Get all ids with resepect to terms/years
cursor.execute(f'SELECT LOWER(lemma), token_id,year FROM embeddings INNER JOIN metadata ON embeddings.File = metadata.workId WHERE LOWER(lemma) = "wolf"')
embeddings.extend([(term, tokenId, year) for term, tokenId, year in cursor.fetchall()])

####
# Aggregation

# Put into data frame
df = pandas.DataFrame.from_dict(embeddings)
df.columns = ["term", "token_id", "year"]
df = df.merge(e, on="token_id", how="inner")  # merge the embeddings

# Get parameters of data
year_min = math.floor(df.year.min()/10)*10 #abrunden damit 1847 = 1840
year_max = math.ceil(df.year.max()/10)*10
intervals = [10]  # SET THIS

r = []
for interval in intervals:
    ranges = numpy.arange(year_min, year_max + interval, interval)  # This creates the year ranges

    # This cuts and groups the df in the wanted intervals per term
    cuts = df.groupby(["term", pandas.cut(df["year"], ranges)]).apply(lambda x: numpy.stack(x["embedding"]))


    # Write an arbitrary pooling function
    def pool(x):
        # r = e[x].mean(0) # Mean pooling
        # r = e[x].sum(0) # SUm pooling ...
        r = x.min(0)  # Minpooling
        return r


    cuts.map(pool)

    result = pandas.concat([cuts.map(pool), df.groupby(["term", pandas.cut(df["year"], ranges)]).agg("count")["year"]],
                           axis=1)
    result.columns = ["embedding", "count"]
    result = result[~result["embedding"].isna()]  # Remove rows where terms do not occurr in the time cut
    result["embedding"] = result["embedding"].map(list)
    r.append(result)

    ## WRITE RESULT
    result.reset_index().to_csv(f"/disk2/ksebestyen/embedding_i{interval}.csv")
