- Este proyecto tiene como dataset   **Fashion MNIST** y para monitoreo la plataforma Weights & Biases
- Este dataset es práctico para poder entrenar modelos de forma rápida en colab
**Conceptos**
- **Data Journey**
    - Es el ciclo de vida a por el cual los datos brutos se transforman en  conocimiento.
    - Es la conexión directa entre el almacenamiento de información con la automatización
    - El siguiente paso es la toma de decisiones inteligentes.
- **Weights & Biases (WB)**
    - Es un aplataforma a través de la cual se puede monitorear en tiempo real el funcionamiento en producción de los modelos, permite trakear versiones
      los cambios de las metricas de rendimiento.

Este modelo pretende simular un modelo implementado en producción de modo que se ha implementado en el porceso la plataforma **WB** para poder reelizar un monitoreo sencillo del proceso de ejecución de los modelos en las que se evidencia:
- Accuracy
- Validation Accuracy
- Loss
- Validation Loss
Adjunto a este reporte se encuentran las gráficas de evidencia y una imagen de pantalla de la plataforma con todas las gráficas.

- Durante el entrenamiento del modelo CNN se evidenció un aumento progresivo de la precisión de entrenamiento y validación, llegando a un 96% y 91% respectivamente.

- La función de perdida disminuyó de forma consistente en el entrenamiento, lo que indica que el modelo aprendió correctamente las características del dataset.

- Sin embargo, pasadas las primeras epocas se evidenció yn leve incremento en la perdida de validación, lo que podría indicar un inicio de overfitting. Esto es una señal de que el modelo ha empezado a especializarse demasiado en los datos de entrenamiento, lo cual reduce la capacidad de generalización.

- Aún con lo anterior, el modelo tuvo un comportamiento general estable y permitió obtener resultados satisfactorios teniendo en cuenta que es una arquitectura CNN básica.

- En una posible mejora se pueden implementar tecnicas como Early Stopping, Dropout y Data Argumentation.