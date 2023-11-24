#-----Importar librerias necesarias
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#-----Entradas y salidas

#damos medidas en pulagas:
entrada = np.array([1, 6, 30, 7, 70, 43, 503, 201, 1005, 99], dtype=float)

#Resultados de la conversión a metros de esas medidas
resultados = np.array([0.0254, 0.1524, 0.762, 0.1778, 1.778, 1.0922, 12.776, 5.1054, 25.527, 2.514], dtype= float)

# Red con 1 capa de entrada y 1 capa de salida
capa1 = tf.keras.layers.Dense(units = 1, input_shape =[1])
# Tipo de red
modelo = tf.keras.Sequential([capa1])

#----------optimizador y metrica de perdida
modelo.compile(
    optimizer= tf.keras.optimizers.Adam(0.1),
    loss = 'mean_squared_error'
)

#Entrenamiento uwu
print("La red esta entrenando uwu")

entrenamiento = modelo.fit(entrada, resultados, epochs=100, verbose= False )

#--------Verificar el entrenamiento de la red

print("ha terminado de entrenar uwu")

#---------Hora de la Predicción uwu 
i = input ("ingresa una medida en pulgadas: ")
i = float (i)

prediccion= modelo.predict([i])
print("La medida que ingresaste en pulgadas su valor en metros es: ", str(prediccion))

#Vizualizar el comportamiento de la red neuronal
plt.xlabel("Ciclos de entrenamiento")
plt.ylabel("Errores")
plt.plot(entrenamiento.history["loss"])
plt.show()
