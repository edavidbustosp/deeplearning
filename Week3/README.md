Este archivo se puede llanar desde google colab.
1. Se realiza cambio del valis del bias se probaron varios valores pero solo con -1.5 y -2 la neurona se comportó como AND
2. Tomando la tendencia de asignar al bias valores cada vez menores (-2.5, -3, -10) las salidas son cero. Con valores cada vez mayores (-1, -0,5, 0, 0.5, 1) las salidas son 1
3. Resultados obtenidos

b=-1.5
bias con valor de: -1.5

Entrada=(0, 0) z=-1.5 salida=> 0
Entrada=(0, 1) z=-0.5 salida=> 0
Entrada=(1, 0) z=-0.5 salida=> 0
Entrada=(1, 1) z=0.5 salida=> 1


b=-2

bias con valor de: -2

Entrada=(0, 0) z=-2.0 salida=> 0
Entrada=(0, 1) z=-1.0 salida=> 0
Entrada=(1, 0) z=-1.0 salida=> 0
Entrada=(1, 1) z=0.0 salida=> 1


b=-2.5

bias con valor de: -2.5

Entrada=(0, 0) z=-2.5 salida=> 0
Entrada=(0, 1) z=-1.5 salida=> 0
Entrada=(1, 0) z=-1.5 salida=> 0
Entrada=(1, 1) z=-0.5 salida=> 0

b=-3

bias con valor de: -3

Entrada=(0, 0) z=-3.0 salida=> 0
Entrada=(0, 1) z=-2.0 salida=> 0
Entrada=(1, 0) z=-2.0 salida=> 0
Entrada=(1, 1) z=-1.0 salida=> 0


b=-1

bias con valor de: -1

Entrada=(0, 0) z=-1.0 salida=> 0
Entrada=(0, 1) z=0.0 salida=> 1
Entrada=(1, 0) z=0.0 salida=> 1
Entrada=(1, 1) z=1.0 salida=> 1


b=-0.5

bias con valor de: -0.5

Entrada=(0, 0) z=-0.5 salida=> 0
Entrada=(0, 1) z=0.5 salida=> 1
Entrada=(1, 0) z=0.5 salida=> 1
Entrada=(1, 1) z=1.5 salida=> 1


b=0

bias con valor de: 0

Entrada=(0, 0) z=0.0 salida=> 1
Entrada=(0, 1) z=1.0 salida=> 1
Entrada=(1, 0) z=1.0 salida=> 1
Entrada=(1, 1) z=2.0 salida=> 1


b=0.5


bias con valor de: 0.5

Entrada=(0, 0) z=0.5 salida=> 1
Entrada=(0, 1) z=1.5 salida=> 1
Entrada=(1, 0) z=1.5 salida=> 1
Entrada=(1, 1) z=2.5 salida=> 1

