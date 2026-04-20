'''
Aplicación de Metricas y Procesamiento de una Red
Neuronal con TensorFlow/Keras
'''

#Carga y Procesamiento

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test)=keras.datasets.mnist.load_data()

X_train=X_train / 255.0
X_test=X_test / 255.0

X_train=X_train.reshape(-1, 28*28)
X_test=X_test.reshape(-1, 28*28)

#Modelo Base sin regularización

def modelo_base():
    print("Construyendo modelo base...")
    model=keras.Sequential([
        keras.layers.Dense(256, activation='relu', input_shape=(784,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model_base=modelo_base()
print("Historia Base")

history_base=model_base.fit(
    X_train, y_train,
    epochs=15,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

print (X_test)
print("--------------------")
print(y_test)
print("--------------------")



'''
Modelo con Regularización
Dropout +L2
'''

print ("Generando modelo regularizado\n\n")

def modelo_regularizado():
    model=keras.Sequential([
        keras.layers.Dense(256, activation='relu', input_shape=(784,),
                           kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(128, activation='relu',
                           kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model_reg=modelo_regularizado()

history_reg=model_reg.fit(
    X_train, y_train,
    epochs=15,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

#Gráficas

#Loss


print("generando gráficas de Loss")

plt.plot(history_base.history['loss'], label='Base - Train')
plt.plot(history_base.history['val_loss'], label='Base - Val')

plt.plot(history_reg.history['loss'], label='Reg - Train')
plt.plot(history_reg.history['val_loss'], label='reg - Val')

plt.title("Comparación de loss")
plt.xlabel("Epocas")
plt.ylabel("Loss")
plt.legend()
plt.show()

#Accuracy

plt.plot(history_base.history['accuracy'], label='Base - Train')
plt.plot(history_base.history['val_accuracy'], label="Base - Val")

plt.plot(history_reg.history['accuracy'], label='Reg - Train')
plt.plot(history_reg.history['val_accuracy'], label='Reg - Val')

plt.title("Comparación de Accuracy")
plt.xlabel("Epocas")
plt.ylabel("Accuracy")
plt.legend()
plt.show()