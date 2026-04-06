9. Análisis y Conclusiones
9.1 Evidencia de overfitting / underfitting
El modelo base (sin regularización) presenta la brecha más amplia entre el loss de entrenamiento y el val_loss: la red memoriza patrones del conjunto de entrenamiento que no se generalizan, lo que se observa en las curvas donde el train_loss converge rápidamente hacia cero mientras el val_loss se estabiliza en un valor mayor o incluso sube. Este comportamiento es la firma clásica del overfitting.

No se observa underfitting en ninguna de las configuraciones, ya que todos los modelos alcanzan métricas de entrenamiento altas (accuracy > 95%).

9.2 Efecto de cada técnica de regularización
Técnica	Mecanismo	Efecto observado
Dropout (0.3)	Desactiva aleatoriamente el 30% de neuronas por batch	Reduce la brecha train-val; curvas más suaves y estables
L2 (λ=0.01)	Penaliza la magnitud de los pesos en la función de pérdida	Fuerza pesos pequeños, evitando que la red dependa de características específicas
Dropout + L2 + Early Stopping	Combinación de las anteriores + detiene el entrenamiento en el mejor punto de val_loss	Mejor generalización global; la red aprende representaciones más robustas
9.3 Hallazgos y dificultades
Hallazgos:

La regularización no siempre mejora el accuracy de test de forma dramática en datasets pequeños y limpios como Breast Cancer; su beneficio principal es la estabilidad y confiabilidad de las predicciones.
El gap entre train_loss y val_loss es el indicador más claro de overfitting; la regularización lo reduce en todos los casos.
Early Stopping actúa como regularización implícita al evitar épocas innecesarias: el modelo combinado puede detenerse antes de la época 80 sin perder rendimiento.
L2 tiende a converger más lentamente que Dropout porque penaliza cada paso de actualización.
Dificultades encontradas:

La elección del hiperparámetro λ en L2 (aquí 0.01) requiere ajuste; valores muy altos pueden causar underfitting.
El valor de patience en Early Stopping (10 épocas) puede detener el entrenamiento prematuramente si hay ruido en la validación.
Con un dataset relativamente pequeño (569 muestras), las diferencias entre modelos pueden ser menores que con datasets más grandes donde el overfitting es más pronunciado.
9.4 Conclusión general
La regularización es una herramienta indispensable para construir modelos que generalicen bien a datos no vistos. En este experimento, la combinación Dropout + L2 + Early Stopping produjo el mejor equilibrio entre loss de validación y accuracy de test, confirmando que las técnicas de regularización son complementarias y su uso conjunto es una práctica recomendada en el entrenamiento de redes neuronales.