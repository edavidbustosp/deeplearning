'''Importación de librerías'''
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

'''Carga de Datos'''
(X_train, y_train),(X_test, y_test)=keras.datasets.mnist.load_data()

#Variables que asignan el valor al Learning Rate
LRATE=0.0001 #Variebla para Learmning Rate bajo
HRATE=0.01 #Variable para learning Rate Alto

#Normalizar

X_train=X_train / 255.0
X_test=X_test / 255.0

#Aplanar

X_train= X_train.reshape(-1, 28*28)
X_test= X_test.reshape(-1, 28*28)

'''Creación del modelo'''

def crear_modelo(learning_rate):
    model=keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    optimizer=keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

''' Experimento
Configuración A 
Learning rate bajo
'''
print ("----------- Configuración A con Learning Rate Bajo -------------")
print("Valor asignaod al Learning Rate ", LRATE)
model_A=crear_modelo(LRATE)

history_A=model_A.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

'''Configuración B
Learning Rate Alto
'''
print ("----------- Configuración B con Learning Rate Alto -------------")
print("Valor asignaod al Learning Rate ", HRATE)
model_B=crear_modelo(HRATE)

history_B=model_B.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

'''Gráficas'''

print("--------- Graficando Loss ---------")

plt.plot(history_A.history['loss'], label='LR='+str(LRATE))
plt.plot(history_B.history['loss'], label='LR='+str(HRATE))

plt.title("Comparación de Loss")
plt.xlabel("Epocas")
plt.ylabel("Loss")
plt.legend()
plt.show()

print("--------- Graficando Accuracy -------")

plt.plot(history_A.history['accuracy'], label='LR='+str(LRATE))
plt.plot(history_B.history['accuracy'], label='LR='+str(HRATE))

plt.title("Comparació de Accuracy")
plt.xlabel("Epocas")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
