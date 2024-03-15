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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, RidgeClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
warnings.filterwarnings('ignore')

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelBinarizer, PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.feature_selection import RFE, SelectFromModel, VarianceThreshold, SelectKBest, r_regression

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

# Espacificación de las variables numéricas
numerical_cols = [
    'temperature',
    'humidity',
    'wind_speed',
    'visibility',
    'solar_radiation',
    'rainfall',
    'snowfall',
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

# Columna objectivo a predecir
target_col = ['rented_bike_count']

# Transformar la columna date al formato de fecha deseado
bikerpro['date'] = pd.to_datetime(bikerpro['date'], format='%d/%m/%Y')

# Crear variable categórica para saber si la fecha fue fin de semana o no
bikerpro['is_weekend'] = np.where(bikerpro['date'].dt.weekday> 4,1,0)
bikerpro['is_weekend'] = bikerpro['is_weekend'].astype('category')

# Crear variable categórica para la hora
bikerpro['hour_as_category'] = bikerpro['hour'].astype('category')

# Crear variable categórica para el día de la semana
bikerpro['weekday'] = bikerpro['date'].dt.weekday.astype('category')

# Datos ordenados
X = bikerpro.sort_values(['date', 'hour'])

# Mostrar resumen de la estructura del archivo
bikerpro.info()

# Datos de entrenamiento
X_train = X.loc[: X.shape[0]-1440,:].drop(target_col, axis=1)
y_train = X.loc[: X.shape[0]-1440,:][target_col]

# Datos de entrenamiento
X_test = X.loc[X.shape[0]-1440+1:,:].drop(target_col, axis=1)
y_test = X.loc[X.shape[0]-1440+1:,:][target_col]

# Complementa los conjuntos de entrenamiento y prueba con las columnas originales y las transformadas a binario
X_train = binary_transform_and_concat(X_train, categorical_binary_cols, categorical_binary_transform_cols)
X_test = binary_transform_and_concat(X_test, categorical_binary_cols, categorical_binary_transform_cols)

# Lista que tiene todas los grupos de columnas
non_target_cols = numerical_cols + categorical_one_hot_cols + categorical_binary_transform_cols

# Definimos los modelos de clasificación que vamos a utilizar
models = [RandomForestClassifier(random_state=42)]

# Definimos las transformaciones que hemos seleccionado previamente
transformers = [
    ('StandardScaler', StandardScaler()),
    ('MinMaxScaler', MinMaxScaler()),
    ('PowerTransformer', PowerTransformer(method='yeo-johnson'))
]

# Definimos los métodos de selección de características que queremos probar
feature_selection_methods = [
    ('SelectKBest', SelectKBest()),
    ('SelectFromModel', SelectFromModel(estimator=RandomForestClassifier())),
    ('RFE', RFE(estimator=RandomForestClassifier()))
]

# Creamos un diccionario con los pipelines para GridSearchCV
pipelines = {}
for model in models:
    for transformer_name, transformer in transformers:
        for selection_method_name, selection_method in feature_selection_methods:
            pipeline_name = f"{model.__class__.__name__}-{transformer_name}-{selection_method_name}"
            pipelines[pipeline_name] = Pipeline([
                ('transformer', transformer),
                ('feature_selection', selection_method),
                ('classifier', model)
            ])

# Definimos los parámetros para GridSearchCV
param_grid = {
    'feature_selection__k': [2, 3],  # Número de características seleccionadas para SelectKBest y RFE
    'classifier__n_estimators': [50, 100, 200]  # Parámetros del clasificador (en este caso, RandomForestClassifier)
}

# Realizamos la búsqueda de hiperparámetros y selección de características
results = {}
for name, pipeline in pipelines.items():
    print(f"Evaluating pipeline: {name}")
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    grid_search.fit(X_train[numerical_cols], y_train)
    results[name] = grid_search.best_score_

# Imprimimos los resultados
best_pipeline = max(results, key=results.get)
print(f"Best pipeline: {best_pipeline}, Best score: {results[best_pipeline]}")