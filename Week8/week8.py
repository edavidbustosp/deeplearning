import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test)=keras.datasets.cifar10.load_data()

#Normalizar

X_train=X_train/255.0
X_test=X_test/255.0
'''
Red Neuronal Convolusional CNN (desde cero)
'''

def modelo_cnn_base():
    model=keras.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        layers.Dense(64,activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

    cnn_base=modelo_cnn_base()

    history_base=cnn_base.fit(
        X_train, y_train,
        epochs=10,
        batch_size=64,
        validation_data=(X_test, y_test)
    )
'''
Transfer Learning
'''


base_model=keras.applications.MobileNetV2(
    input_shape=(32,32,3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable=False

model_tl=keras.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model_tl.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_tl=model_tl.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test)
)

'''
Graficación
'''
plt.plot(history_base.history['accuracy'], label='CNN base')
plt.plot(history_tl.history['accuracy'], label='Transfer Learning')

plt.title("Comparación de Accuracy")
plt.xlabel("Epocas")
plt.ylabel("Accuracy")
plt.legend()
plt.show()