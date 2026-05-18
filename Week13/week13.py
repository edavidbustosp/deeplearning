'''
########################
Importar Librerías
########################
'''


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt


factor=0.3
epocas=50

'''
############################
Cargar Dataset
############################
'''

(X_train, _), (X_test, _)=keras.datasets.fashion_mnist.load_data()

'''
#############################
Normalización
#############################
'''

X_train=X_train.astype("float32")/255.0
X_test=X_test.astype("float32")/255.0


'''
##############################
Aplanar imágenes
##############################
'''


X_train=X_train.reshape(-1, 784)
X_test=X_test.reshape(-1, 784)


'''
##############################
Adición de Ruido Artificial
##############################
'''


X_train_noisy=X_train+factor*np.random.normal(
    loc=0.0,
    scale=1.0,
    size=X_train.shape
)

X_test_noisy=X_test+factor*np.random.normal(
    loc=0.0,
    scale=1.0,
    size=X_test.shape


)
#Limitar Valores

X_train_noisy=np.clip(X_train_noisy, 0., 1.)
X_test_noisy=np.clip(X_test_noisy, 0., 1.)


'''
############################
Autodecoder
############################
'''

#Encoder
#Comprime la información

input_img=keras.Input(shape=(784,))

encoded=layers.Dense(128, activation='relu')(input_img)
encoded=layers.Dense(64, activation='relu')(encoded)
encoded=layers.Dense(32, activation='relu')(encoded)

#Decoder
#Reconstruye la imagen

decoded=layers.Dense(64, activation='relu')(encoded)
decoded=layers.Dense(128, activation='relu')(decoded)
decoded=layers.Dense(784, activation='sigmoid')(decoded)

#Modelo Completo

autoencoder=keras.Model(input_img, decoded)


'''
#############################
Compilación
#############################
'''

autoencoder.compile(
    optimizer='adam',
    loss='binary_crossentropy'
)

'''
#############################
Entrenamiento
#############################
'''

history = autoencoder.fit(
    X_train_noisy,
    X_train,
    epochs=epocas,
    batch_size=256,
    shuffle=True,
    validation_data=(X_test_noisy, X_test)
)

'''
#############################
Reconstrucción
#############################
'''

decoded_imgs=autoencoder.predict(X_test_noisy)

'''
##############################
Visualización Comparativa
##############################
'''

n=5

plt.figure(figsize=(12,6))

for i in range(n):
    #Original
    ax=plt.subplot(3, n, i + 1)
    plt.imshow(X_test[i].reshape(28,28), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    #Ruido
    ax=plt.subplot(3, n, i + 1 + n)
    plt.imshow(X_test_noisy[i].reshape(28,28), cmap="gray")
    plt.title("Ruido")
    plt.axis("off")

    #Reconstruida

    ax=plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28,28), cmap="gray")
    plt.title("Reconstruida")
    plt.axis("off")

plt.show()


'''
#########################
Gráfica de Loss
#########################
'''

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.xlabel("Epocas")
plt.ylabel("Loss")
plt.legend()
plt.title("Comportamiento del entrenamiento")

plt.show()
