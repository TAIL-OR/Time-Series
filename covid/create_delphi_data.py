import sqlite3
import pandas as pd
from columns import data_cols
from prettytable import from_db_cursor
import sys
from time import sleep
import re
import subprocess

subprocess.run(['python3', 'createdb.py'])

def remover_acentos(texto):
    substituicoes = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u',
        'â': 'a', 'ê': 'e', 'î': 'i', 'ô': 'o', 'û': 'u',
        'ã': 'a', 'õ': 'o',
        'ç': 'c',
        'ñ': 'n',
        'ü': 'u', 'ï': 'i', 'ë': 'e', 'ö': 'o', 'ä': 'a',
        'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
        'À': 'A', 'È': 'E', 'Ì': 'I', 'Ò': 'O', 'Ù': 'U',
        'Â': 'A', 'Ê': 'E', 'Î': 'I', 'Ô': 'O', 'Û': 'U',
        'Ã': 'A', 'Õ': 'O',
        'Ç': 'C',
        'Ñ': 'N',
        'Ü': 'U', 'Ï': 'I', 'Ë': 'E', 'Ö': 'O', 'Ä': 'A'
    }

    padrao = re.compile('|'.join(substituicoes.keys()))

    resultado = padrao.sub(lambda m: substituicoes[m.group(0)], texto)

    return resultado

def clean_filename(name):
    # Remover caracteres especiais e substituir espaços por underscores
    cleaned_name = ''.join(c if c.isalnum() or c in [' ', '_'] else '_' for c in name)
    cleaned_name = remover_acentos(cleaned_name)
    return cleaned_name.replace('/', ' ').replace('2', 'II').lower().replace(' ', '_')

populations = pd.read_csv('../data/df_locs.csv')
populations = populations[['loc', 'population']]
populations['loc'] = populations['loc'].apply(lambda x: x.lower())
populations['loc'] = populations['loc'].apply(lambda x: x.replace(' ', '_'))
populations['loc'] = populations['loc'].apply(lambda x: clean_filename(x))

ras_with_pop = populations[populations['population'].isnull() == False]['loc'].tolist()
ras_no_pop = populations[populations['population'].isnull() == True]['loc'].tolist()

# Conecte ao banco de dados
conn = sqlite3.connect('../data/covid_data.db')

# Crie um cursor
c = conn.cursor()

# Obtenha a consulta do arquivo query.sql
query = open('delphi_query.sql', 'r').read()

# Execute a consulta
c.execute(query)

table = from_db_cursor(c)

# Transforme o resultado da consulta em um DataFrame
df = pd.read_sql_query(query, conn)
other_cols = ['new_hospitalization', 'total_hospitalization', 'new_icu', 'total_icu', 'new_recovered', 'total_recovered']
# add those columns and set them to null
for col in other_cols:
    df[col] = None

ras = []
#207595100.84125105,0.0002265492688374578,0.00013604804959844592,0.590502482908539,0.6973610108787711,0.1254817776180646,7.571127412581456e-06,2.839172779718048e-07,1.6088645751735588e-06,55315.15757741388,2447.587454130462,1732.9247515883887,0.00553596077726755,7.09793194929512e-08,489.51749082609234,11552.831677255928, South America,

model_state_df = pd.DataFrame(columns=['S', 'E', 'I', 'UR', 'DHR', 'DQR', 'UD', 'DHD', 'DQD', 'R', 'D', 'TH', 'DVR', 'DVD', 'DD', 'DT', 'continent', 'country', 'province'])
ras_with_no_100cases = []

ras_to_exclude = ['entorno_df']

for _, group_df in df.groupby(['country', 'province']):
    country_name, province_name = _

    # Remover caracteres especiais e substituir espaços por underscores
    country_name = clean_filename(country_name)
    province_name = clean_filename(province_name)
    
    if province_name not in ras_to_exclude and group_df['day_since100'].max() > 0:

        ras.append(province_name)

        filename = f'../danger_map/processed/Global/Cases_{country_name}_{province_name}.csv'
        group_df['day_since100'] = group_df['day_since100'].apply(lambda x: int(x) if pd.notnull(x) else None)
        group_df['province'] = group_df['province'].apply(lambda x: remover_acentos(x.replace(' ', '_')).lower() if pd.notnull(x) else None)

        group_df.to_csv(filename, index=False)

        try:
            model_state_df = model_state_df.append({'S': populations[populations['loc'] == province_name]['population'].values[0]
                                                    ,'E': 0.0002265492688374578, 'I': 0.00013604804959844592, 
                                                    'UR': 0.590502482908539, 'DHR': 0.6973610108787711, 'DQR': 0.1254817776180646, 
                                                    'UD': 7.571127412581456e-06, 'DHD': 2.839172779718048e-07, 'DQD': 1.6088645751735588e-06, 
                                                    'R': 55315.15757741388, 'D': 2447.587454130462, 'TH': 1732.9247515883887, 
                                                    'DVR': 0.00553596077726755, 'DVD': 7.09793194929512e-08, 
                                                    'DD': 489.51749082609234, 'DT': 11552.831677255928, 'continent': 'South America', 
                                                    'country': country_name, 'province': province_name}, ignore_index=True)
        except:
            print(f"Error with {province_name}")

        if group_df['case_cnt'].max() < 100:
            ras_with_no_100cases.append(province_name)
    else:
        print(f"Excluding {province_name}")

print("RAs with no 100 cases: ", ras_with_no_100cases)

model_state_df.to_csv('/home/franky/Time-Series/DELPHI/data_sandbox/predicted/raw_predictions/Predicted_model_state_V4_2020-10-01.csv', index=False)

df.to_csv('../data/delphi_data.csv', index=False)

ras_in_pop = set(ras) & set(ras_with_pop)
ras_not_in_pop = set(ras) - set(ras_with_pop)

print("RAs with no population: ", ras_not_in_pop)

population_df = pd.DataFrame(columns=['Continent', 'Country', 'Province', 'pop2016'])

for ra in ras_in_pop:
    population_df = population_df.append({'Province': ra, 'pop2016': int(populations[populations['loc'] == ra]['population'].values[0])}, ignore_index=True)

population_df['Continent'] = 'South America'
population_df['Country'] = 'brasilia'

population_df.to_csv('../danger_map/processed/Global/Population_Global.csv', index=False)

conn.close()