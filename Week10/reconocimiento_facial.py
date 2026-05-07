#Importación de Librerías

import cv2
import numpy as np
import face_recognition
import tensorflow as tf
import matplotlib.pyplot as plt


img_loaded=face_recognition.load_image_file('/home/david/Nextcloud/UDEC/Especializacion IA/Deep Learning/Git/deeplearning/Week10/media/mandela.jpg')
img_loaded=cv2.cvtColor(img_loaded, cv2.COLOR_BGR2RGB)
#--------- Detectando Rostro ----------------------
face=face_recognition.face_locations(img_loaded)[0]


#Codificanco Imagen

train_enconde=face_recognition.face_encodings(img_loaded)[0]


test=face_recognition.load_image_file('/home/david/Nextcloud/UDEC/Especializacion IA/Deep Learning/Git/deeplearning/Week10/media/mandelaJoven.jpg')
test=cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
test_encode=face_recognition.face_encodings(test)[0]

resultado=face_recognition.compare_faces([train_enconde], test_encode)

print("VALIDACIÓN DE COINCIDENCIA===>>> ", resultado)

cv2.rectangle(img_loaded, (face[3], face[0]), (face[1], face[2]), (255,0,255), 1)
cv2.imshow('Imagen Cargada', img_loaded)
cv2.waitKey(0)

plt.subplot(1,2,1)
plt.imshow(img_loaded)
plt.title("Imagen cargada")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(test)
plt.title("Imagen de comparación")
plt.axis("off")

if resultado[0]:

    plt.suptitle(
        f"Similitud de Imágenes - Son la misma persona"        
    )
else:
    plt.suptitle(
        f"Similitud de Imágenes - No son la misma persona"        
    )

plt.show()