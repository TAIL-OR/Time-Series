import pandas as pd
import plotly.express as px
import os

def get_plots():
    if not os.path.exists('data/query_result.csv'):
        print("Generating CSV file...")
        make_command = os.system('make -f GenerateData')

        if make_command != 0:
            print("Was not possible to generate the CSV file.")
            print("Trying to create the database...")

            create_command = os.system('python3 createdb.py')

            if create_command != 0:
                print("Was not possible to create the database.")
                exit(1)

    df = pd.read_csv('data/query_result.csv')

    fig = px.line(df, x='week', y='total', color='ra', markers=True, title='Total de Casos por Semana em Cada Região',
                  labels={'week': 'Semana', 'total': 'Total de Casos'})

    fig.update_xaxes(type='category')

    # Salvar o gráfico como um arquivo PNG
    fig.write_html("covid_plot.html")
    
    print("Plot saved as covid_ts.png")

if __name__ == '__main__':
    get_plots()