import pandas as pd
import sqlite3

# Read the data
df = pd.read_csv('../data/VRA_20240119005511.csv', sep=';')
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

conn = sqlite3.connect('../data/flights_data.db')
df.to_sql('flights_data', conn, if_exists='replace', index=False)


conn.commit()
conn.close()
