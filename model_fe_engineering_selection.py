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
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import PolynomialFeatures

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import r_regression

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer

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

# Crear variable categórica para el número de mes en la fecha
bikerpro['month'] = bikerpro['date'].dt.month.astype('category')

# Crear variable categórica para saber si la fecha fue fin de semana o no
bikerpro['is_weekend'] = np.where(bikerpro['date'].dt.weekday> 4,1,0)
bikerpro['is_weekend'] = bikerpro['is_weekend'].astype('category')

# Crear variable categórica para la hora
bikerpro['hour_as_category'] = bikerpro['hour'].astype('category')

numerical_cols = [
    'temperature',
    'humidity',
    'wind_speed',
    'visibility',
    'dew_point_temperature',
    'solar_radiation',
    'rainfall',
    'snowfall',
 ]

# Realizar una copia de las columnas numéricas originales
bikerpro_numerical = bikerpro[numerical_cols].copy()

# Normalizar los datos numéricos utilizando StandardScaler
scaler = StandardScaler()
bikerpro_numerical = scaler.fit_transform(bikerpro_numerical)

# Aplicar la transformación Yeo-Johnson a las columnas numéricas copiadas
yeo_johnson = PowerTransformer(method='yeo-johnson')
bikerpro_numerical = yeo_johnson.fit_transform(bikerpro_numerical)

# Crear nombres de columnas para las nuevas columnas transformadas
numerical_cols_transform = [f'{col}_transf' for col in numerical_cols]

# Crear un nuevo DataFrame con las columnas numéricas originales y las transformadas
bikerpro = pd.concat([bikerpro, pd.DataFrame(bikerpro_numerical, columns=numerical_cols_transform).astype('category')], axis=1)

# Datos ordenados
X = bikerpro.sort_values(['date', 'hour'])

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

# Datos de entrenamiento
X_train = X.loc[: X.shape[0]-1440,:].drop(target_col, axis=1)
y_train = X.loc[: X.shape[0]-1440,:][target_col]

# Datos de entrenamiento
X_test = X.loc[X.shape[0]-1440+1:,:].drop(target_col, axis=1)
y_test = X.loc[X.shape[0]-1440+1:,:][target_col]

# Define listas de columnas que van a emplearse en el modelado
weather_cols = [
    'temperature',
    'humidity',
    'wind_speed',
    'visibility',
    'dew_point_temperature',
    'solar_radiation',
    'rainfall',
    'snowfall',
 ]

seasons_cols = [
    'seasons',
    'holiday',
    'functioning_day',
    'month',
    'hour_as_category',
    'temperature_transf',
    'humidity_transf',
    'wind_speed_transf',
    'visibility_transf',
    'dew_point_temperature_transf',
    'solar_radiation_transf',
    'rainfall_transf',
    'snowfall_transf'
]

time_cols = ['hour']

# Lista que tiene todas los grupos de columnas
non_target_cols = weather_cols + seasons_cols + time_cols


# Pipeline para escalar con estandar z-score
numerical_pipe = Pipeline([
    ('standar_scaler', StandardScaler()),
    # ----------- Aqui seleccionamos las 4 mejores variables -------- #
    ('select_k_best',SelectKBest(r_regression, k=4) ),
])

# Pipeline para aplicar one hot encoding
categorical_pipe = Pipeline([
    ('one_hot', OneHotEncoder(handle_unknown='ignore'))
])

# Combina ambos procesos en columnas espeficadas en listas
pre_processor = ColumnTransformer([
    ('numerical', numerical_pipe, weather_cols),
    ('categorical', categorical_pipe, seasons_cols),
], remainder='passthrough')

# comunica al pipeline la lista en el orden que se deben aplicar
# estos pasos

pipe_standard_ohe = Pipeline([
    ('transform', pre_processor),
    ('model', KNeighborsRegressor(n_neighbors=5))
])

# Realiza la transformacion de los datos y el ajuste del modelo
pipe_standard_ohe.fit(X_train[non_target_cols], y_train)

y_train_pred = pipe_standard_ohe.predict(X_train[non_target_cols])
y_test_pred = pipe_standard_ohe.predict(X_test[non_target_cols])

# error en conjunto de entrenamiento y prueba
error_train = root_mean_squared_error(y_train, y_train_pred)
error_test = root_mean_squared_error(y_test, y_test_pred)

# errores
print("Error RSME en train:", round(error_train,2) )
print("Error RSME en test:", round(error_test,2) )