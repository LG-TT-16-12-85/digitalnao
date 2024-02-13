# Dependencias importantes:
# pip install -U ipykernel
# pip install pandas seaborn
# pip install pyarrow
# pip install scikit-learn

# Importar dependencias
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Importar clase de Python para instanciar el modelo KNN
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error

# Se definen las variables que contienen la ruta del archivo
DATA_PATH="C:\\Users\\TTLG-85\\Documents\\DigitalNAO\\ID1"
FILE_BIKERPRO = 'SeoulBikeData.csv'

# Leemos el archivo con pandas
bikerpro = pd.read_csv(
    os.path.join(DATA_PATH, FILE_BIKERPRO),
    encoding = "ISO-8859-1"
    )

# Se asignan los nombres de las columnas limpias en una variable
clean_columns = [
    x.lower().\
        replace("(°c)", '').\
        replace("(%)", '').\
        replace(" (m/s)", '').\
        replace(" (10m)", '').\
        replace(" (mj/m2)", '').\
        replace("(mm)", '').\
        replace(" (cm)", '').\
        replace(" ", '_')
    for x in bikerpro.columns
    ]

# Ahora asignamos los nuevos nombres de columnas para el análisis
bikerpro.columns = clean_columns

# Transformar la columna date al formato de fecha deseado
bikerpro['date'] = pd.to_datetime(bikerpro['date'], format='%d/%m/%Y')

# Mostrar resumen de la estructura del archivo
bikerpro.info()

# Columnas del clima
weather_cols = [
    'temperature', 
    'humidity',
    'humidity',
    'wind_speed',
    'visibility',
    'dew_point_temperature',
    'solar_radiation',
    'rainfall',
    'snowfall'
    ]

# Columna objectivo a predecir
target_col = ['rented_bike_count']

# Obtenemos los datos requeridos para el modelo
X = bikerpro[weather_cols+target_col]

# datos ordenados
X = bikerpro.sort_values(['date', 'hour'])

# Datos de entrenamiento
X_train = X.loc[: X.shape[0]-1440,:][weather_cols]
y_train = X.loc[: X.shape[0]-1440,:][target_col]

# Datos de prueba
X_test = X.loc[X.shape[0]-1440+1:,:][weather_cols]
y_test = X.loc[X.shape[0]-1440+1:,:][target_col]

# KNN con RMSE
# Definir el arreglo de los valores de k
knn_list = [3, 5, 10, 15, 20, 50, 100, 300, 500, 1000]

# Listas para valores del RMSE de entremiento y prueba para los distintos valores de k 
rmse_train = []
rmse_test = []

# Iterar sobre el arreglo y evaluar el modelo
for k in knn_list:
    # Modelo para k vecinos mas cercanos
    model = KNeighborsRegressor(n_neighbors=k)
    # Ajustamos el modelo sobre los datos de entrenamiento
    model.fit(X_train, y_train)
    # Predecimos para los datos de entrenamiento y prueba
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    # error en conjunto de entrenamiento
    error_train = root_mean_squared_error(y_train, y_train_pred)
    # agregar el valor del rmse de train en su arreglo
    rmse_train.append(error_train)
    # error en conjunto de prueba
    error_test = root_mean_squared_error(y_test, y_test_pred)
    # agregar el valor del rmse de test en su arreglo
    rmse_test.append(error_test)
    print("Error RMSE en ENTRENAMIENTO:", round(error_train,2) )
    print("Error RMSE en PRUEBA:", round(error_test,2) )
    # Plotting graph of Actual (true) values and predicted values
    plt.figure(figsize=(15,5))
    plt.plot(y_train.reset_index(drop=True))
    plt.plot(list(y_train_pred))
    plt.title('Datos de entrenamiento cuando k es igual a: {}'.format(k))
    plt.legend(["Actual", "Predicted"])
    plt.show()
    # Plotting graph of Actual (true) values and predicted values
    plt.figure(figsize=(15,5))
    plt.plot(y_test.reset_index(drop=True))
    plt.plot(list(y_test_pred))
    plt.title('Datos de prueba cuando k es igual a: {}'.format(k))
    plt.legend(["Actual", "Predicted"])
    plt.show()

# Visualizar cómo cambia el RMSE con diferentes valores de k para el conjunto de entrenamiento y prueba
plt.plot(knn_list, rmse_train, marker='o', label='Entrenamiento')
plt.plot(knn_list, rmse_test, marker='o', label='Prueba')
plt.title('RMSE vs. Valor de k')
plt.xlabel('Valor de k')
plt.ylabel('RMSE')
plt.xticks(knn_list)
plt.legend()
plt.grid(True)
plt.show()
