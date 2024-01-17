import sqlite3
import pandas as pd
from columns import data_cols
from prettytable import from_db_cursor
import sys

# Conecte ao banco de dados
conn = sqlite3.connect('./data/covid_data.db')

# Crie um cursor
c = conn.cursor()

# Obtenha a consulta do arquivo query.sql
query = open('qrst.sql', 'r').read()

# Execute a consulta
c.execute(query)

table = from_db_cursor(c)
print(table[:10])

# Transforme o resultado da consulta em um DataFrame
df = pd.read_sql_query(query, conn)

# Salve como CSV se o argumento for "save"

df.to_csv('data/query_result.csv', index=False)
print("Resultado da consulta salvo como CSV.")

# Feche a conexão
conn.close()