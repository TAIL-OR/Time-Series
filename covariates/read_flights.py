import sqlite3
import pandas as pd
from prettytable import from_db_cursor

conn = sqlite3.connect('../data/flights_data.db')
c = conn.cursor()
query = open('f_query.sql', 'r').read()
c.execute(query)

df = pd.read_sql_query(query, conn)

df.to_csv('../data/flights_results.csv', index=False)
