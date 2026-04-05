Para este ejercicio se tomó como modelo el dataset load_digits de sklearn
1- Se realiza la importación de las respectivas librerías icluyendo las de graficación
2- Carga de los datos del dataset
3- Se manejón un array de taza de aprendizaje de 0.0001, 0.001 y 0.01
4- El resultado respecto a las tazas de aprendizaje fue que conforme se aumentó la taza de aprendizaje (learning rate) el modelo tuvo mejor rendimiento.
    LR= 0.0001 Accurancy:  0.9666666666666667
    LR= 0.001 Accurancy:  0.9694444444444444
    LR= 0.01 Accurancy:  0.9722222222222222
5- En este resultado se identificó una limitante, ya que se itentó cargar un learning rate de 0.1 y se generó una inestabilidad en el aprendizaje que hizo que
    el rendimiento bajara. De los resultados se agregan las gráficas.
    LR= 0.1 Accurancy:  0.9388888888888889
6- En la sgráficas agregadas se evidencia que entre más enpinada la gráfica es equivalente a un aprendizaje más rápido
7- Para la prueba con los optimizadores se utilizaron
    -SGD, ADAM, LBFGS
8- Resultados de la prueba con optimizadores
    -Optimizador=SGD Accurancy:  0.9583333333333334
    Optimizador=ADAM Accurancy:  0.9694444444444444
    Optimizador=LBFGS Accurancy:  0.9583333333333334
9- Se evidencia que con ADAM fue el que presentó mejor desempeño de los 3 ya que SGD y LBFGS tuvieron el mismo desempeño y menor a ADAM

La taza de aprendizaje influye en la convergencia del modelo con lo cual se evidenció que se alcanzan mejres valores en un número menor de iteraciones.
El optimizador ADAM presentó el mejor desempeño de los 3, y según lo investigado este tiene capacidad de ajustar de forma dinámica la taza de aprendizaje mientras 
transcurre el entrenamiento lo que se traduce en una mejor convergencia.
