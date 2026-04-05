import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Carga de Datos


data=load_diabetes()
X=data.data
y=data.target

# Entrenamiento / Test

#Test Size 20% es decir prueba el 80% restante es entrenamiento 

#De aquí se cargan x en Entrenamiento y x en test igualmente sucede con y, y se ejecuta la funcion train_test_split()
#En los argumentos se cargar X e y, el tamaño de test, el random state  y la clave.
X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.2, random_state=7
)

#Escalado
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#Modelo baseline (regresión lineal)

from sklearn.linear_model import LinearRegression

baseline=LinearRegression()
baseline.fit(X_train, y_train)

pred_b=baseline.predict(X_test)

print("Modelo Base MSE: ", mean_squared_error(y_test, pred_b))
print("Modelo Base R2: ", r2_score(y_test, pred_b))

#Red Neuronal Perceptrón Multi Capa (MLPRegressor)

from sklearn.neural_network import MLPRegressor

mlp=MLPRegressor(
    hidden_layer_sizes=(32,),
    activation="relu",
    solver="adam",
    max_iter=2000,
    random_state=7
    
)
mlp.fit(X_train, y_train)

pred=mlp.predict(X_test)

print("MLP MSE: ", mean_squared_error(y_test, pred)) #Error cuadrático, entre más bajo mejor

'''Para R2
1=ok
0=No Util
<0= No funciona
'''

print("MLP R2: ", r2_score(y_test, pred))
