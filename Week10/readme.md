En este ejercicio se implementa una apriximación funcional de una Red Neuronal Siamesa
para reconocimiento facial

Es básicamente un modelo preentrenado utilizando la librería
face_recognition que permite hacer el reconocimiento de rostros
Esto facilita el ejercicio ya que evita utilizar datasets publicos
que pueden no contener suficientes imágenes de lo que se le pueda cargar

En este ejercicio básicamente se realizan las siguientes actividades
- Detectar los rostros presentes en la imagen
- Extrae las caracteristicas que definen cada rostro
- Compara dichas características
- Determina si hay similitud facial

para el ejercicio de colab se adicionó la posibilidad de 
cargar las imagenes a comparar de forma manual a través de la 
librería files de google.colab.

Al cargar las imágenes se dibuja un rectangulo en el rostro de la persona 
y luego se realizan las comparaciones.

Se muestran las imágenes con los resultados de comparación 
y se indica si son o no la misma persona.