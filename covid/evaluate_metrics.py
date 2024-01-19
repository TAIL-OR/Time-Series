import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error

model_names = ['cnn_model', 'lstm_model', 'dnn_model', 'lstm_cnn_model', 'skip_cnn_lstm_model']

def get_dataframe(model_name):
    df = pd.read_csv(f'../data/{model_name}_predictions.csv')
    df['model'] = model_name
    return df

def get_all_dataframes():
    all_data =  [get_dataframe(model_name) for model_name in model_names]
    return all_data

def get_metrics(df):
    mse = mean_squared_error(df['actual'], df['prediction'])
    mae = mean_absolute_error(df['actual'], df['prediction'])
    return mse, mae

def get_all_metrics():
    all_data = get_all_dataframes()
    all_metrics = [get_metrics(df) for df in all_data]
    
    metrics = pd.DataFrame(all_metrics, columns=['mse', 'mae'])
    metrics['model'] = model_names
    return metrics

metrics_df = get_all_metrics()
print(metrics_df)