Para el modelo de diabetes, el dataset utilizado está basado en cifras
1- El dataset utilizado es load_diabetes de sklearn
2- Se asigna un 20% para pruebas o validaciones de las predicciones y el 80% para entrenamiento
3- Se realiza el escalado para los modelos bases de el error cuadrático medio MSE y para la regresión lineal R2
4- Con base en los modelos base se procede a realizar la regresión del Preceptrón Multi Capa MLP
5- Esta red Neuronal tiene una capa oculta de 32 neuronas y con esa configuración se realizaron las pruebas.
6- Deacuerdo con los datos del modelo base, se espera que las predicciones supere el 40% n la variabilidad.
7- En el ejercicio realizado se obtuvo apenas un 32% lo cual quiere decir que el modelo aunque logra aprender, no lo logra como se espera 
    Luego entonces no es un modelo que logre superar el modelo base y por lo tanto no hace un aporte significativo.