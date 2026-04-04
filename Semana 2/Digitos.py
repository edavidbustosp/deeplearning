'''
Programa para la clasificación de numeros escritor a mano utilizando el dataset digits de sklearn

'''

#Importación de librerías necesarias

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

#Carga de los datos

data=load_digits()
X=data.data
y=data.target
iteraciones=3000

#Split
print("Datos \n", X)
'''
------ Separación Entrenamiento - Prueba --------

Test Size 20% es decir prueba el 80% restante es entrenamiento 

De aquí se cargan x en Entrenamiento y x en test igualmente sucede con y, y se ejecuta la funcion train_test_split()
En los argumentos se cargar X e y, el tamaño de test, el random state  y la clave.
'''

X_train, X_test, y_train, y_test=train_test_split(
    X, y, test_size=0.1, random_state=7
)

#----------- Escalado de Datos --------------

scaler=StandardScaler()
X_train_s=scaler.fit_transform(X_train)
X_test_s=scaler.transform(X_test)


'''
------------- Modelo Base ------------
Regresión Logística

### Exlicar los que se muestra en laimpresión de los datos ######
'''
print("Train: ", X_train_s.shape, "Test:", X_test_s.shape)


''' 
--------- Regresión Logística --------------------
Carga de la variable baseline (Modelo Base) con la regresión Logística 
Se establece el valor máximo de iteraciones
Llamado de la función fit y se le envían como argumentos X_train_s y y_train
'''

from sklearn.linear_model import LogisticRegression
baseline=LogisticRegression(max_iter=iteraciones)
baseline.fit(X_train_s, y_train)

'''
------- Predicción ----------
A la variable baseline (Modelo Base) invocando la función predict se le envía X_test_s
-------- Metrica ---------

'''

pred_b=baseline.predict(X_test_s)
#Porcentaje de aciertos 

print("Precisión de Referencia: ", accuracy_score(y_test, pred_b))

#Perceptrón Multi Capa MLP

from sklearn.neural_network import MLPClassifier

mlp=MLPClassifier(
    hidden_layer_sizes=(64,32),
    activation="relu",
    max_iter=iteraciones,
    random_state=7
)

mlp.fit(X_train_s, y_train)
pred=mlp.predict(X_test_s)

print("Resultado con ", iteraciones , " Iteraciones")
print("Precisión del MLP: ", accuracy_score(y_test, pred))
