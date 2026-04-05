'''
Demostraciones que se deben hacer
1. ¿Cómo cambia el aprendizaje?
2. ¿Cómo influyen los hiperparametros?
3. ¿Cómo infloyen los optimizadores?
4. Evidencias
'''

#Importaciómn de librerías

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

#Carga de Datos

data=load_digits()
X=data.data
y=data.target
iteraciones=2000

#Split

'''
Entrenamiento con una asignación para test de 20%
'''

X_train, X_test, y_train, y_test=train_test_split(
    X, y, test_size=0.2, random_state=7) 

#Escalado
scaler=StandardScaler()
X_train_s=scaler.fit_transform(X_train)
X_test_s=scaler.transform(X_test)

'''
Taza de aprendizaje
Primer Experimento

'''

tazas_aprendizaje=[0.0001, 0.001, 0.01]

for lr in tazas_aprendizaje:
    mlp=MLPClassifier(
        hidden_layer_sizes=(64,),
        learning_rate_init=lr,
        max_iter=iteraciones,
        random_state=7
    )

    mlp.fit(X_train_s, y_train)
    pred=mlp.predict(X_test_s)

    print(f"LR= {lr} Accurancy: ", accuracy_score(y_test, pred))

    plt.plot(mlp.loss_curve_, label=f"lr{lr}")

plt.title("Evolusión de perdida según taza de aprendizaje")
plt.xlabel("Iteraciones")
plt.ylabel("Perdida")
plt.legend()
plt.show()

'''
Optimizadores
Segundo Experimento
'''

optimizadores=["sgd", "adam", "lbfgs"]

for optimizador in optimizadores:
    mlp=MLPClassifier(
        hidden_layer_sizes=(64,),
        solver=optimizador,
        max_iter=iteraciones,
        random_state=7
    )
    mlp.fit(X_train_s, y_train)
    pred=mlp.predict(X_test_s)

    print(f"Optimizador={optimizador.upper()} Accurancy: ",accuracy_score(y_test, pred))

    if hasattr(mlp, "loss_curve_"):
        plt.plot(mlp.loss_curve_, label=optimizador.upper())

plt.title("Comparación de Optimizadores")
plt.xlabel("Iteraciones")
plt.ylabel("loss")
plt.legend()
plt.show()

