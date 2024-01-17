import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from keras.models import Sequential
from keras import Model
from keras.layers import Dense, LSTM, Flatten, InputLayer, Conv1D, GRU, Dropout, MaxPooling1D
from keras.layers import Concatenate, Input
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.losses import MeanSquaredError, Huber
from keras.optimizers import Adam
from keras.models import load_model
from keras.backend import clear_session
from keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
import plotly.express as px

dataframe = pd.read_csv('data/query_result.csv')
dataframe['week'] = dataframe['week'].str.strip()
dataframe['date'] = pd.to_datetime(dataframe['week'] + '-1', format='%Y-%W-%w')
dataframe = dataframe.set_index('date')
dataframe = dataframe.drop(['week'], axis=1)
dataframe = dataframe.sort_index()

df_pivoted = dataframe.pivot_table(index='date',columns='ra', values='total', fill_value=0, aggfunc='sum')
df_pivoted.reset_index(inplace=True)

final_df = df_pivoted.set_index('date')

def df_to_X_y(df, window_size=5):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np) - window_size):
        row = [r for r in df_as_np[i:i+window_size]]
        X.append(row)
        label = df_as_np[i+window_size]
        y.append(label)

    return np.array(X), np.array(y)


def plot_predictions(model, X, y, file='plot_result'):
    predictions = model.predict(X)
    num_variables = y.shape[1]

    # Criar DataFrame com as variáveis reais e preditas
    df = pd.DataFrame(y, columns=[f'Actual_{i+1}' for i in range(num_variables)])
    df_pred = pd.DataFrame(predictions, columns=[f'Prediction_{i+1}' for i in range(num_variables)])
    df = pd.concat([df, df_pred], axis=1)

    # Plotar com Plotly Express
    fig = px.line(df, labels={'value': 'Value', 'variable': 'Variable'},
                  title='Actuals vs Predictions for Multiple Variables')
    fig.update_layout(legend_title_text='Legend')
    fig.write_html(f'data/{file}.html')

    return df, mse(y, predictions)

def get_params():
    lr = 5e-4
    epochs = 1000
    horizon = 30
    batch_size = 32
    return lr, epochs, horizon, batch_size

model_config = {}

def cfg_model_run(model, history, test_ds):
    return {'model': model, 'history': history, 'test_ds': test_ds}

WINDOW_SIZE = 5
X, y = df_to_X_y(final_df, window_size=WINDOW_SIZE)
print(X.shape, y.shape)

X_train, y_train = X[:-30], y[:-30]
X_test, y_test = X[-30:], y[-30:]

print(X_train.shape, y_train.shape)

INPUT_SHAPE = X_train.shape[1:]
OUTPUT_SHAPE = y_train.shape[1:]

print(INPUT_SHAPE, OUTPUT_SHAPE)

# First -> DNN Model

print(get_params()[0])

def dnn_model():
    clear_session()
    
    model = Sequential([
        InputLayer(input_shape=INPUT_SHAPE),
        Flatten(),
        Dense(128, 'relu'),
        Dense(128, 'relu'),
        Dense(64, 'relu'),
        Dense(OUTPUT_SHAPE[0], 'linear')
    ], name='dnn_model')

    loss = Huber()
    optimizer = Adam(learning_rate=get_params()[0])
    model.compile(loss=loss, optimizer=optimizer, metrics=['mae', 'mse'])

    return model

# Second -> CNN Model

def cnn_model():
    clear_session()

    model = Sequential([
        Conv1D(64, 6, activation='relu', padding='causal', input_shape=INPUT_SHAPE),
        MaxPooling1D(2),
        Conv1D(64, 3, activation='relu', padding='causal'),
        MaxPooling1D(2),
        Flatten(),
        Dense(64, 'relu'),
        Dense(OUTPUT_SHAPE[0], 'linear')

    ], name='cnn_model')

    loss = Huber()
    optimizer = Adam(learning_rate=get_params()[0])
    model.compile(loss=loss, optimizer=optimizer, metrics=['mae', 'mse'])

    return model

# Third -> LSTM Model

def lstm_model():
    clear_session()

    model = Sequential([
        LSTM(72, return_sequences=True, input_shape=INPUT_SHAPE, activation='relu'),   
        LSTM(48, return_sequences=False, activation='relu'),
        Flatten(),
        Dense(128, 'relu'),
        Dense(OUTPUT_SHAPE[0], 'linear')
    ], name='lstm_model')

    loss = Huber()
    optimizer = Adam(learning_rate=get_params()[0])
    model.compile(loss=loss, optimizer=optimizer, metrics=['mae', 'mse'])

    return model

# Fourth -> CNN-LSTM Model

def lstm_cnn_model():
    clear_session()

    model = Sequential([
        Conv1D(64, 6, activation='relu', padding='causal', input_shape=INPUT_SHAPE),
        MaxPooling1D(2),
        Conv1D(64, 3, activation='relu', padding='causal'),
        MaxPooling1D(2),
        LSTM(72, return_sequences=True, activation='relu'),
        LSTM(64, return_sequences=False, activation='relu'),
        Flatten(),
        Dense(128, 'relu'),
        Dense(64, 'relu'),
        Dense(OUTPUT_SHAPE[0], 'linear')
    ], name='lstm_cnn_model')

    loss = Huber()
    optimizer = Adam(learning_rate=get_params()[0])
    model.compile(loss=loss, optimizer=optimizer, metrics=['mae', 'mse'])

    return model

# Fifth -> LSTM-CNN Model SkipConnection

def skip_cnn_lstm_model():
    clear_session()

    inputs = Input(shape=INPUT_SHAPE, name='main')
    conv1 = Conv1D(64, 6, activation='relu', padding='causal')(inputs)
    max1 = MaxPooling1D(2)(conv1)
    conv2 = Conv1D(64, 3, activation='relu', padding='causal')(max1)
    max2 = MaxPooling1D(2)(conv2)
    lstm1 = LSTM(72, return_sequences=True, activation='relu')(max2)
    lstm2 = LSTM(48, return_sequences=False, activation='relu')(lstm1)
    flat = Flatten()(lstm2)

    skip_flat = Flatten()(inputs)

    concat = Concatenate(axis=-1)([flat, skip_flat])
    dense1 = Dense(128, 'relu')(concat)
    dense2 = Dense(64, 'relu')(dense1)
    output = Dense(OUTPUT_SHAPE[0], 'linear')(dense1)

    model = Model(inputs=inputs, outputs=output, name='skip_cnn_lstm_model')
    
    loss = Huber()
    optimizer = Adam(learning_rate=get_params()[0])
    model.compile(loss=loss, optimizer=optimizer, metrics=['mae', 'mse'])

    return model


def run_model(model_name, model_function, model_configs, epochs):
    model = model_function()

    checkpoint = ModelCheckpoint(f'models/{model_name}.h5', save_best_only=True)

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=get_params()[3], validation_split=0.2, callbacks=[checkpoint])

    test_ds = model.evaluate(X_test, y_test)

    model_configs[model_name] = cfg_model_run(model, history, test_ds)

    return 'Model Trained'

# run models

run_model('dnn_model', dnn_model, model_config, get_params()[1])
run_model('cnn_model', cnn_model, model_config, get_params()[1])
run_model('lstm_model', lstm_model, model_config, get_params()[1])
run_model('lstm_cnn_model', lstm_cnn_model, model_config, get_params()[1])
run_model('skip_cnn_lstm_model', skip_cnn_lstm_model, model_config, get_params()[1])

legend = list()

fig, axs = plt.subplots(1,5, figsize=(15, 10))

def plot_metrics(metric, val, ax, upper, file_name, model_name):
    ax.plot(val['history'].history[metric])
    ax.plot(val['history'].history[f'val_{metric}'])
    
    ax.legend([f'Train {metric}', f'Val {metric}'])
    ax.set_xlabel('epochs')
    ax.set_ylabel(metric)

    ax.set_ylim([0, upper])

    ax.set_title(f'{model_name} {metric}')

    fig.savefig(f'data/{model_name}_{file_name}.png')

    plt.close(fig)

def print_metrics(model_name, val):
    print(f'{model_name} - loss: {val["test_ds"][0]} - mae: {val["test_ds"][1]} - mse: {val["test_ds"][2]}')

for (key, val), ax in zip(model_config.items(), axs.flatten()):
    print_metrics(key, val)
    legend.append(key)
    
for (key, val) in model_config.items():
    best_model = load_model(f'models/{key}.h5')

    y_pred = best_model.predict(X_test)

    df, mse_value = plot_predictions(val['model'], X_test, y_test, key)
    df.to_csv(f'data/{key}_predictions.csv')
    
    mae = np.mean(np.abs(y_test - y_pred))
    print(f'{key} - MAE: {mae}')

    # save metrics in text
    with open(f'data/txt_metrics/{key}_metrics.txt', 'w') as f:
        f.write(f'{key} - MSE: {mse_value} - MAE: {mae}')