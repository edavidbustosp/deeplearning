"""
=========================================================
RED SIAMESA PARA RECONOCIMIENTO FACIAL
USANDO FACE_RECOGNITION
=========================================================

Autor: David Bustos Pulido

Descripción:
Este proyecto implementa una aproximación funcional
de una Red Neuronal Siamesa para reconocimiento facial.

La lógica consiste en:

1. Detectar rostros
2. Extraer embeddings faciales
3. Comparar embeddings
4. Determinar similitud facial

=========================================================
"""

# =====================================================
# IMPORTACIÓN DE LIBRERÍAS
# =====================================================

import cv2
import numpy as np
import face_recognition
import matplotlib.pyplot as plt

# =====================================================
# FUNCIÓN PARA CARGAR IMAGEN
# =====================================================

def cargar_imagen(path):

    """
    Carga una imagen y extrae:

    - imagen
    - ubicación del rostro
    - encoding facial
    """

    # -------------------------------------------------
    # CARGAR IMAGEN
    # -------------------------------------------------

    image = face_recognition.load_image_file(path)

    # -------------------------------------------------
    # DETECCIÓN DE ROSTRO
    # -------------------------------------------------

    faces = face_recognition.face_locations(image)

    if len(faces) == 0:

        raise ValueError(
            f"No se detectó rostro en {path}"
        )

    # -------------------------------------------------
    # EXTRAER EMBEDDING
    # -------------------------------------------------

    encoding = face_recognition.face_encodings(image)[0]

    return image, faces[0], encoding

# =====================================================
# FUNCIÓN PARA DIBUJAR RECTÁNGULO
# =====================================================

def dibujar_rostro(image, face):

    top, right, bottom, left = face

    cv2.rectangle(
        image,
        (left, top),
        (right, bottom),
        (255,0,255),
        2
    )

    return image

# =====================================================
# RUTAS DE IMÁGENES
# =====================================================

image1_path = "/home/david/Nextcloud/UDEC/Especializacion IA/Deep Learning/Git/deeplearning/Week10/media/mandela.jpg"

image2_path = "/home/david/Nextcloud/UDEC/Especializacion IA/Deep Learning/Git/deeplearning/Week10/media/mandelaJoven.jpg"

# =====================================================
# CARGA DE IMÁGENES
# =====================================================

print("===================================")
print("CARGANDO IMÁGENES")
print("===================================")

img1, face1, encode1 = cargar_imagen(
    image1_path
)

img2, face2, encode2 = cargar_imagen(
    image2_path
)

# =====================================================
# COMPARACIÓN FACIAL
# =====================================================

print("\n===================================")
print("COMPARANDO ROSTROS")
print("===================================")

resultado = face_recognition.compare_faces(
    [encode1],
    encode2
)

# =====================================================
# DISTANCIA ENTRE EMBEDDINGS
# =====================================================

distance = face_recognition.face_distance(
    [encode1],
    encode2
)

print("Distancia facial:", distance[0])

# =====================================================
# RESULTADO
# =====================================================

print("\n===================================")
print("RESULTADO")
print("===================================")

if resultado[0]:

    mensaje = "MISMA PERSONA"

    print("Resultado:", mensaje)

else:

    mensaje = "PERSONAS DIFERENTES"

    print("Resultado:", mensaje)

# =====================================================
# DIBUJAR ROSTROS
# =====================================================

img1 = dibujar_rostro(img1, face1)
img2 = dibujar_rostro(img2, face2)

# =====================================================
# VISUALIZACIÓN
# =====================================================

plt.figure(figsize=(10,5))

# -----------------------------------------------------
# IMAGEN 1
# -----------------------------------------------------

plt.subplot(1,2,1)

plt.imshow(img1)

plt.title("Imagen Base")

plt.axis("off")

# -----------------------------------------------------
# IMAGEN 2
# -----------------------------------------------------

plt.subplot(1,2,2)

plt.imshow(img2)

plt.title("Imagen de Comparación")

plt.axis("off")

# -----------------------------------------------------
# TÍTULO GENERAL
# -----------------------------------------------------

plt.suptitle(
    f"""
    Resultado: {mensaje}

    Distancia Facial:
    {distance[0]:.4f}
    """
)

plt.show()