import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar o arquivo CSV
df1 = pd.read_csv('../data/query_result.csv')
df1 = df1.groupby('week').sum().reset_index()
df2 = pd.read_csv('../data/flights_results.csv')

df = pd.merge(df1, df2, on='week')

print(df.head())

# Plotar a correlação entre as colunas 'total' e 'flights'
plt.scatter(df['total'], df['flights'])
plt.title(f'Gráfico de Dispersão entre Casos e Voos')
plt.xlabel('total')
plt.ylabel('flights')
correlacao = df['total'].corr(df['flights'])

plt.savefig('../plots/plot_correlation.png')