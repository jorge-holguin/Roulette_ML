import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Función para cargar los datos desde un archivo de texto
def cargar_datos(nombre_archivo):
    with open(nombre_archivo, 'r') as file:
        # Split each line by comma and flatten the list of lists into a single list
        data = [int(number) for line in file for number in line.strip().split(',')]
    return np.array(data)
15

# Función para preparar los datos en ventanas de tamaño window_size
def preparar_datos(data, window_size):
    X = np.array([data[i:i + window_size] for i in range(len(data) - window_size)])
    y = data[window_size:]
    return X, y

# Función para añadir el último número al dataset y actualizar el modelo
def actualizar_modelo(ultimo_numero, data, window_size=3):
    # Añadir el último número al conjunto de datos
    data_actualizado = np.append(data, ultimo_numero)

    # Preparar los datos con el conjunto actualizado
    X, y = preparar_datos(data_actualizado, window_size)

    # Crear y entrenar el modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, data_actualizado

# Función para predecir el número más probable después del último número ingresado
def predecir_numero(model, data, ultimo_numero, window_size=3):
    # Usar los últimos window_size - 1 números y el último número ingresado
    secuencia = np.append(data[-(window_size-1):], ultimo_numero).reshape(1, -1)
    resultado_futuro = model.predict(secuencia)[0]
    return resultado_futuro

# Nombre del archivo donde se encuentran los números
nombre_archivo = "C:\\Users\\jorge\\Documents\\Proyectos\\numeros.txt"

# Cargar datos desde el archivo de texto
data = cargar_datos(nombre_archivo)

# Inicialización del modelo
window_size = 3
model, data = actualizar_modelo(data[0], data, window_size)  # Inicia con el primer dato como ejemplo

while True:
    # Solicitar al usuario el último número que salió
    ultimo_numero = int(input("Ingresa el último número que salió en la ruleta (o -1 para salir): "))

    # Verificar si el usuario quiere salir
    if ultimo_numero == -1:
        break

    # Actualizar el modelo con el nuevo número
    model, data = actualizar_modelo(ultimo_numero, data, window_size)

    # Predecir el número más probable después del último número ingresado
    resultado_futuro = predecir_numero(model, data, ultimo_numero, window_size)

    # Mostrar resultado
    print(f"El número más probable después del número {ultimo_numero} es: {resultado_futuro}")
