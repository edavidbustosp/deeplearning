Este ejercicio consiste en la impleentación de una red GAN
No es una red que funcione de forma perfecta pero de acuerdo
con la teoría etudiada, se implementó el generador y el discriminador

Dentro de las practicas para poder entrenar el generador fue necesario congelar el discriminador
para permitir el entrenamiento del generador.

El modelo se probó inicialmente con 1.000 epocas pero se evidenció que las imágenes no eran claras
Se le aumentó a 10.000 y aunque cambiaron un poco las imágenes seguían si ser del todo claras

Otro ejercicio que se hizo en cuanto a infraestructura fue cambiar el modo de ejecución a GPU en colab
esto debido a que una GAN entrena lento y consume bastantes recursos