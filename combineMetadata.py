#workId
#Year
import sqlite3
import pandas
columns = ["workId", "Year"]
metadataCSV = pandas.read_csv("/disk2/ksebestyen/extCompDB.csv", sep=",", usecols=columns)
#
# embed_db = sqlite3.connect('/disk2/ksebestyen/embed_db_with_file.db', detect_types=sqlite3.PARSE_DECLTYPES)
# cursor = embed_db.cursor()
#
# cursor.execute('''DROP TABLE IF EXISTS metadata''')
#
# sql_ = '''CREATE TABLE metadata (
#         workId TEXT,
#         year INTEGER
# )'''
# cursor.execute(sql_)

for idx in metadataCSV.index:
    data = metadataCSV.loc[idx, :].values.tolist()
    print(data)
#     sql_ = '''INSERT INTO metadata values (?, ?)'''
#     cursor.execute(sql_, tuple([data[0], data[1]]))
#
# embed_db.commit()
# print("embeddings saved in database")

# for column in ["workId", "year"]:
#     sql_ = f'''CREATE INDEX {column} on metadata ({column})'''
#     cursor.execute(sql_)
#
# embed_db.commit()
#
# embed_db.close()
