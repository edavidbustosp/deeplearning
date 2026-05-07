import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test)=keras.datasets.cifar10.load_data()

#Normalizar

X_train=X_train/255.0
X_test=X_test/255.0
