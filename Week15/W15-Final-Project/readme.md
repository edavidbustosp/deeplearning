En este proyecto se implementa un modelo CNN para **clasificación automática de flores.**

El objetivo es aplicar conceptos de:
- Deep Learning
- Monitoreo de entrenamiento
- Optimización de Modelos
- Trazabilidad experimental

Para este ejercicio se utiliza TensorFlow/keras y Weights  6 Biases

Durante la ejecuciión del proyecto se aplicó la conexión con Weights & Biases plataforma en la cual se pueden visualizar las métricas de ejeución del modelo
Igualmente se utilizaron funciones como:
- Conv2D que permite extraer características de las imágenes de flores como bordes, colores, formas y pétalos.
- MxPooling que reduce las dimensiones y mejora la eficiencia
- Dense que realiza la clasificación final
- Dropout que ayuda con la reducción de overfitting apagando neuronas aleatoriamente.

Para temas de monitoreo
Waights & Biases permitió:
- Registro de Métricas de forma automática
- Comparar los Experimentos
- Analizar el comportamiento del entrenamiento
- Detectar Overfitting
- Visualizar accuracy y loss en tiempo real.

Con este modelo se logró que la red neuronal aprendiera las características visualas más relevantes en flores
Este tipo de modelos se pueden aplicar en:

- Agricultura inteligente
- Clasificación de flores y plantas
- En la industria del sector floricultor
