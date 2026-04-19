Se generaron dos modelos

El learning rate básicamemte nos ayuda a indicarle al modelo que tan grandes con los pasos que debe dar durante el aprendizaje

Medelo A con Learning Rate bajo  de 0.0001
- En este punto los pasos son pequeños
- El modeo aprende lento
- Los cambios en los pesos son pequeños
- En general es un entrenamiento estable, tiene más precisión y es menos propenso a desestabilizarse

Con un LR bajo se pued edecir que el modelo tarda mucho en el entrenamiento y puede no terminar el entrenamiento.



Modelo B con Learning Rate Alto de 0.01

- Aquín los pasos son más grandes
- Consecuentemente  el aprendizaje es mucho más rápido
- Los cambios en los pesos son grandes
- Tarda menos en entrenamiento y converge rápido.

Debido a la velocidad puede presentar inestabilidad y puede presetnar rebotes  en lugar de curvas suavizadas.

En este ejemplo se configuraron dos variables a las cuales se les puede ajustar el valor y de este modo se puede validar como varía el aprendizaje del modelo

LRATE= Variebla para Learmning Rate bajo
HRATE= Variable para learning Rate Alto