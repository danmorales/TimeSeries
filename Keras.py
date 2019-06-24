import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import os

def criar_arranjo(dados, window_size = 1):
    dado_X, dado_Y = [], []
    for i in range(len(dados) - window_size - 1):
        a = dados[i:(i + window_size), 0]
        dado_X.append(a)
        dado_Y.append(dados[i + window_size, 0])
    return(np.array(dado_X), np.array(dado_Y))
    
def modelo_keras(treino_X, treino_Y, window_size = 1):
    modelo = Sequential()
    modelo.add(LSTM(32, input_shape = (1, window_size)))
    modelo.add(Dense(1))
    modelo.summary()
    modelo.compile(loss = "mean_squared_error", optimizer = "adam")
    modelo.fit(treino_X, treino_Y, epochs = 20, batch_size = 1, verbose = 2)
    
    return(modelo)

def previsao_e_score(modelo, X, Y):
    previsao = scaler.inverse_transform(modelo.predict(X))
    dado_original = scaler.inverse_transform([Y])
    score = math.sqrt(mean_squared_error(dado_original[0], previsao[:, 0]))
    return(score, previsao)

def previsao_futuro(modelo,X):
   previsao = scaler.inverse_transform(modelo.predict(X))
   return(previsao)

dateparse = lambda dates: pd.datetime.strptime(dates, '%d-%m-%Y %H:%M')
pd_dados = pd.read_csv('Train_SU63ISt.csv',parse_dates=['Datetime'], index_col='Datetime',date_parser=dateparse)
data_arranjo = pd_dados.index
pd_dados = pd_dados.reset_index()
pd_dados = pd_dados.drop("Datetime",axis=1)
pd_dados = pd_dados.drop("ID",axis=1)

dados = pd_dados.values.astype("float32")

scaler = MinMaxScaler(feature_range=(0, 1))
dados = scaler.fit_transform(dados)

tamanho_treino = 0.60

tamanho_treino = int(len(dados)*tamanho_treino)
tamanho_teste = len(dados)-tamanho_treino

treino, teste = dados[0:tamanho_treino, :], dados[tamanho_treino:len(dados), :]

print("Numero de entradas (arranjo treino, arranjo teste): " + str((len(treino), len(teste))))

window_size = 1

treino_X, treino_Y = criar_arranjo(treino, window_size)
teste_X, teste_Y = criar_arranjo(teste, window_size)
print("Forma do arranjo de dados de treino:")
print(treino_X.shape)
print("Forma do arranjo de dados de teste:")
print(teste_X.shape)

treino_X = np.reshape(treino_X, (treino_X.shape[0], 1, treino_X.shape[1]))
teste_X = np.reshape(teste_X, (teste_X.shape[0], 1, teste_X.shape[1]))
print("Nova forma do arranjo treino:")
print(treino_X.shape)
print("Nova forma do arranjo teste:")
print(teste_X.shape)

modelo = modelo_keras(treino_X, treino_Y, window_size)

rmse_treino, previsao_treino = previsao_e_score(modelo, treino_X, treino_Y)
rmse_teste, previsao_teste = previsao_e_score(modelo, teste_X, teste_Y)

print("Training data score: %.2f RMSE" % rmse_treino)
print("Test data score: %.2f RMSE" % rmse_teste)

grafico_previsao_treino = np.empty_like(dados)
grafico_previsao_treino[:, :] = np.nan
grafico_previsao_treino[window_size:len(previsao_treino) + window_size, :] = previsao_treino

grafico_previsao_teste = np.empty_like(dados)
grafico_previsao_teste[:, :] = np.nan
grafico_previsao_teste[len(previsao_treino) + (window_size * 2) + 1:len(dados) - 1, :] = previsao_teste

plt.figure(figsize = (15, 5))
plt.plot(data_arranjo,scaler.inverse_transform(dados), label = "Valor verdadeiro")
plt.plot(data_arranjo,grafico_previsao_treino, label = "Previsão dados treino")
plt.plot(data_arranjo,grafico_previsao_teste, label = "Previsão dados teste")
plt.xlabel("Meses")
plt.ylabel("Número de passageiros")
plt.title("Comparando dados verdadeiros e previstos")
plt.legend()
plt.show()

savefig('Keras.png',dpi=100)    

#pd_futuro = pd.read_csv('Test_0qrQsBZ.csv',parse_dates=['Datetime'], index_col='Datetime',date_parser=dateparse)
#data_arranjo_futuro = pd_futuro.index