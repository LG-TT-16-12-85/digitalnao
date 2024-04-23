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
import pickle

import warnings
from sklearn.ensemble import RandomForestRegressor
warnings.filterwarnings('ignore')

from sklearn.metrics import root_mean_squared_error
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.feature_selection import SelectKBest, r_regression

# Función para transformar las columnas especificadas de un Data Frame
# Regresa un dataframe con las nuevas columnas
# completeDataFrame = Es el Data Frame roiginal
# columnNamesTransform = Son las columnas a transformar
# categorical_binary_transform_cols = Son los nombres de las nuevas columnas transformadas (_binary)
def binary_transform_and_concat(completeDataFrame, columnNamesTransform, categorical_binary_transform_cols):
    # Inicializar el data frame con las columnas a transformar
    categorical_binary_df = pd.DataFrame(completeDataFrame[columnNamesTransform])
    # Inicializar el LabelBinarizer
    lb = LabelBinarizer()
    # Aplicar LabelBinarizer a cada columna
    for col in categorical_binary_df.columns:
        categorical_binary_df[f'{col}_binary'] = lb.fit_transform(categorical_binary_df[col])
    # Concatenar las nuevas columnas con el Data Frame original
    finalDataFrame = pd.concat([completeDataFrame, categorical_binary_df[categorical_binary_transform_cols]], axis=1)
    
    return finalDataFrame

# Se definen las variables que contienen la ruta del archivo
DATA_PATH="C:\\Users\\Admin\\Documents\\academic\\digitalnao\\bikepro"
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

# Crear variable categórica para saber si la fecha fue fin de semana o no
bikerpro['is_weekend'] = np.where(bikerpro['date'].dt.weekday> 4,1,0)
bikerpro['is_weekend'] = bikerpro['is_weekend'].astype('category')

# Crear variable categórica para la hora
bikerpro['hour_as_category'] = bikerpro['hour'].astype('category')

# Crear variable categórica para el día de la semana
bikerpro['weekday'] = bikerpro['date'].dt.weekday.astype('category')

# Características de rezagos (lags)
bikerpro['rented_bike_lag_1'] = bikerpro['rented_bike_count'].shift(1).fillna(0)  # Bicicletas rentadas en la hora anterior
bikerpro['rented_bike_lag_8'] = bikerpro['rented_bike_count'].shift(8).fillna(0)  # Bicicletas rentadas en las pasadas 8 horas
bikerpro['rented_bike_lag_16'] = bikerpro['rented_bike_count'].shift(16).fillna(0)  # Bicicletas rentadas en las pasadas 16 horas
bikerpro['rented_bike_lag_24'] = bikerpro['rented_bike_count'].shift(24).fillna(0)  # Bicicletas rentadas en las últimas 24 horas

# Datos ordenados
X = bikerpro.sort_values(['date', 'hour'])

# Mostrar resumen de la estructura del archivo
bikerpro.info()

# Columna objectivo a predecir
target_col = ['rented_bike_count']

# Espacificación de las variables numéricas
numerical_cols = [
    'temperature',
    'humidity',
    'wind_speed',
    'visibility',
    'solar_radiation',
    'rainfall',
    'snowfall',
    'rented_bike_lag_1',
    'rented_bike_lag_8',
    'rented_bike_lag_16',
    'rented_bike_lag_24'
]

# Especificación de las variables categóricas utilizando one-hot
categorical_one_hot_cols = [
    'seasons',
    'hour_as_category',
    'weekday'
]

# Especificación de las variables categóricas binarias
categorical_binary_cols = [
    'holiday',
    'functioning_day',
    'is_weekend'
]

# Especificación de las variables categóricas binarias transformadas
categorical_binary_transform_cols = [
    'holiday_binary',
    'functioning_day_binary',
    'is_weekend_binary'
]

# Datos de entrenamiento
X_train = X.loc[: X.shape[0]-1440,:].drop(target_col, axis=1)
y_train = X.loc[: X.shape[0]-1440,:][target_col]

# Datos de prueba
X_test = X.loc[X.shape[0]-1440+1:,:].drop(target_col, axis=1)
y_test = X.loc[X.shape[0]-1440+1:,:][target_col]

# Complementa los conjuntos de entrenamiento y prueba con las columnas originales y las transformadas a binario
X_train = binary_transform_and_concat(X_train, categorical_binary_cols, categorical_binary_transform_cols)
X_test = binary_transform_and_concat(X_test, categorical_binary_cols, categorical_binary_transform_cols)

# Lista que tiene todas los grupos de columnas
non_target_cols = numerical_cols + categorical_one_hot_cols + categorical_binary_transform_cols

X_train = X_train[non_target_cols]
X_test = X_test[non_target_cols]

# Pipeline para transformación de características numéricas
numerical_pipe = Pipeline([
    ('numerical_transformer', StandardScaler()),
    ('selection', SelectKBest(r_regression, k=3))
])

# Pipeline para transformación de características categóricas
categorical_pipe = Pipeline([
    ('categorical_transformer', OneHotEncoder(handle_unknown='ignore'))
])

# Combina ambos procesos en columnas espeficadas en listas
pre_processor = ColumnTransformer([
    ('numerical', numerical_pipe, numerical_cols),
    ('categorical', categorical_pipe, categorical_one_hot_cols),
], remainder='passthrough')

# Se comunica al pipeline la lista en el orden que se deben aplicar
pipe_transform_model = Pipeline([
    ('transform', pre_processor),
    ('model', RandomForestRegressor(
        max_features=0.76475,
        n_estimators=79,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=49,
        oob_score=False,
        max_depth=10,
        min_samples_split=11,
        max_leaf_nodes=61))
])

# Ejecutar la búsqueda de parámetros en tus datos
pipe_transform_model.fit(X_train, y_train)

y_train_pred = pipe_transform_model.predict(X_train)
y_test_pred = pipe_transform_model.predict(X_test)

# Error en conjunto de entrenamiento y prueba
error_train = root_mean_squared_error(y_train, y_train_pred)
error_test = root_mean_squared_error(y_test, y_test_pred)

# Imprime el valor del RMSE para el mejor modelo
print("Error RSME en train:", round(error_train,2))
print("Error RSME en test:", round(error_test,2))

pickle.dump(pipe_transform_model, open('model_prediction_bikerpro.pkl', 'wb'))