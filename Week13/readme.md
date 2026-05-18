Actividad 13: Implementación de un Autoencoder en una Red Denoising usando el dataset MNIST en Google Colab
En esta actividad se utiliza el dataset fashion-MNIST que tiene dentro de su colección imágenes de zapatos, bolsos camisetas etc.

Lo que se busca con este modelo es la demostración de:

encoder
decoder
compresión
reconstrucción
eliminación de ruido
aprendizaje no supervisado
Conceptos Importantes
* Autoencoder
Red Neuronal que comprime información, aprende representación interna y reconstruye los datos originales.

* Denoising Autoencoder
Es un modelo entrenado para eliminar ruidos de las imágenes

A las imágenes originales se les aplicó ruido para el entrenamiento de modo que el modelo eliminara los pixeles de ruido
y generara una imágen de salida.
en la gráfica final se evidencia que a medida que transcurren las epocas el loss va bajando
de este modo se puede decir que el aprendizaje es estable
Las imágenes reconstruidas conservaron la estructura original en el contorno y forma de estas
Si se observó que en algunas hubo una lijera perdida de detalle especialmente la camiseta que tenía letras algo borrosas
en la imagen original, por lo que al no distinguirse perfectamente es posible que el modelo
haya tomado esos pixeles como ruido.
