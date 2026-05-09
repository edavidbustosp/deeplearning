
'''
PASO 1

--------------------------
IMPORTACIÓN D ELIBRERÍAS
--------------------------
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


epocas=20

'''
paso 2

------------------------------
GENERACIÓN DE SEREIE TEMPORAL
------------------------------

Creación de dataset secuencial
'''

time=np.arange(0, 500)

series=np.sin(0.1 * time) + np.random.normal(scale=0.1, size=len(time))

'''
------------------
Visualización de la serie
------------------
'''

plt.figure(figsize=(12,5))

plt.plot(series)

plt.title("Serie Temporal")

plt.show()


'''

PASO 3

-------------------------
Normalizaciómn de Datos
-------------------------
'''

scaler=MinMaxScaler()

series_scaled=scaler.fit_transform(
    series.reshape(-1,1)
)

'''

PASO 4

-------------------------------
Crear Ventanas Temporales
-------------------------------
'''

#===========================
#FUNCIÓN PARA CREACIÓN DE SECUENCIAS
#===========================

def create_sequences(data, seq_lenght):
    X=[]
    y=[]

    for i in range(len(data) - seq_lenght):
        X.append(
            data[i:i+seq_lenght]
        )

        y.append(
            data[i+seq_lenght]
        )
    return np.array(X), np.array(y)

#===========================
#CREACIÓN DE SECUENCIAS
#===========================

SEQ_LENGTH=20

X, y=create_sequences(
    series_scaled,
    SEQ_LENGTH
)

print(X.shape)
print(y.shape)

'''

PASO 5

----------------------
Entrenamiento/Prueba
----------------------


'''

split=int(len(X) * 0.8)

X_train=X[:split]
X_test=X[split:]

y_train=y[:split]
y_test=y[split:]

print(X_train.shape)

'''

PASO 6

-----------------------
MODELO BASE LSTM
-----------------------

baseline
'''


model_lstm=keras.Sequential([

    layers.LSTM(
        64,
        input_shape=(SEQ_LENGTH,1)
    ),

    layers.Dense(1)
])

#==========================
#COMPILAR
#==========================

model_lstm.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

#===========================
#ENTENAR
#===========================

history_lstm=model_lstm.fit(

    X_train,
    y_train,

    epochs=epocas,
    batch_size=32,

    validation_data=(X_test, y_test)
)


'''

PASO 7

----------------------
VISUALIZACIÓN DE LOSS
----------------------

'''

plt.plot(history_lstm.history['loss'])

plt.plot(history_lstm.history['val_loss'])

plt.title("Loss - LSTM")

plt.legend("Train", "Validation")

plt.show()

'''

PASO 8

-----------------------
MODELO DE ATENCIÓN
-----------------------

'''

#=====================
#ENTRADA
#=====================

inputs=keras.Input(
    shape=(SEQ_LENGTH,1)
)

#===========================
#LSTM
#===========================

lstm=layers.LSTM(
    64,
    return_sequences=True
)(inputs)

#=============================
#MECANISMO DE ATENCIÓN
#=============================

attention=layers.Attention()(
    [lstm, lstm]
)

#==============================
#GLOBAL POOLING
#==============================

pool=layers.GlobalAveragePooling1D()(
    attention
)

#===============================
#SALIDA
#===============================

outputs=layers.Dense(1)(pool)

#===============================
#MODELO
#===============================

model_attention=keras.Model(
    inputs,
    outputs
)

#===============================
#COMPILAR
#===============================

model_attention.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

#===============================
#ENTRENAMIENTO
#===============================

history_attention=model_attention.fit(

    X_train,
    y_train,

    epochs=epocas,

    batch_size=32,
    validation_data=(X_test,y_test)
)

'''

PASO 9

--------------------
MINI TRANSFORMER
--------------------

'''
#=================================
#ENTRADA
#=================================

inputs=keras.Input(
    shape=(SEQ_LENGTH,1)
)

#===================================
#MULTI HEAD ATTENTION
#===================================

attention=layers.MultiHeadAttention(
    num_heads=2,
    key_dim=32
)(
    inputs,
    inputs
)

#====================================
#NORMALIZACIÓN
#====================================

x=layers.LayerNormalization()(
    attention
)

#=====================================
#FLATEN
#=====================================

x=layers.Flatten()(x)

#======================================
#DENSE
#======================================

x=layers.Dense(
    64,
    activation='relu'
)(x)

#======================================
#SALIDA
#======================================

outputs=layers.Dense(1)(x)

#======================================
#MODELO
#======================================

model_transformer=keras.Model(
    inputs,
    outputs
)

#======================================
#COMPILAR
#======================================

model_transformer.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

#=======================================
#ENTRENAMIENTO
#=======================================

history_transformer=model_transformer.fit(

    X_train,
    y_train,
    epochs=epocas,
    batch_size=32,
    validation_data=(X_test, y_test)
)


'''

PASO 10

-------------------------
COMPARACIÓN DE MODELOS
-------------------------

'''

#========================================
#EVALUACIÓN
#========================================

print("LSTM")
print(
    model_lstm.evaluate(X_test,y_test)
)

print("\nAttention")
print(
    model_attention.evaluate(X_test, y_test)
)

print("\nTransformer")
print(
    model_transformer.evaluate(X_test, y_test)
)


'''

PASO 11

---------------------
VISUALIZACIÓN
---------------------

'''

pred_lstm=model_lstm.predict(X_test)

pred_attention=model_attention.predict(X_test)

pred__transformer=model_transformer.predict(X_test)

#=========================================
#GRÁFICA
#=========================================

plt.figure(figsize=(14,6))

plt.plot(y_test, label='Real')

plt.plot(pred_lstm, label='LSTM')

plt.plot(pred_attention, label='Attention')

plt.plot(pred__transformer, label="Transformer")

plt.legend()

plt.title("Comparación de Predicciones")

plt.show()