import pandas as pd
import plotly.express as px
from time import sleep
import os

def get_plots():
    if not os.path.exists('../data/query_result.csv'):
        print("Generating CSV file...")
        os.system('python3 createdb.py')
        sleep(1)
        make_command = os.system('make -f GenerateData')
        if make_command != 0:
            print("Error while generating data.")
            exit()

    df = pd.read_csv('../data/query_result.csv')

    fig = px.line(df, x='week', y='total', color='ra', markers=True, title='Total de Casos por Semana em Cada Região',
                  labels={'week': 'Semana', 'total': 'Total de Casos'})

    fig.update_xaxes(type='category')

    # Salvar o gráfico como um arquivo PNG
    fig.write_html("covid_plot.html")
    print("Plot saved as covid_ts.png")
    

if __name__ == '__main__':
    get_plots()