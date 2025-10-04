# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import zipfile      
import pickle       
import gzip         
import json         
import os           
import pandas as pd 
from sklearn.model_selection import GridSearchCV       
from sklearn.decomposition import PCA           
from sklearn.pipeline import Pipeline                    
from sklearn.compose import ColumnTransformer    
from sklearn.neural_network import MLPClassifier        
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix         
from sklearn.preprocessing import OneHotEncoder, StandardScaler  
from sklearn.feature_selection import SelectKBest, f_classif    



def limpiar(df):
    # Creamos una copia del DataFrame para no modificar el original
    df = df.copy()
    
    # Eliminamos la columna ID ya que no aporta información predictiva
    df = df.drop('ID', axis=1)
    
    # Renombramos la columna objetivo para facilitar su manejo posterior
    df = df.rename(columns={'default payment next month': 'default'})
    
    # Eliminamos registros con valores faltantes (NaN)
    df = df.dropna()
    
    # Filtramos registros con valores no válidos (0 = N/A) en EDUCATION y MARRIAGE
    df = df[(df['EDUCATION'] != 0 ) & (df['MARRIAGE'] != 0)]
    
    # Agrupamos niveles superiores de educación (>4) en la categoría "others" (valor 4)
    df.loc[df['EDUCATION'] > 4, 'EDUCATION'] = 4

    return df

def modelo():
    # Definimos las variables categóricas que necesitan codificación one-hot
    # SEX: género, EDUCATION: nivel educativo, MARRIAGE: estado civil
    categoricas = ['SEX', 'EDUCATION', 'MARRIAGE']  
    
    # Definimos las variables numéricas que necesitan escalado
    # Incluye límite de crédito, edad, historial de pagos y montos
    numericas = [
        "LIMIT_BAL",  # Límite de crédito
        "AGE",        # Edad
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",  # Historial de pagos
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",  # Montos a pagar
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5","PAY_AMT6"  # Montos pagados
    ]

    # Configuramos el preprocesador que aplicará diferentes transformaciones
    # a distintos tipos de variables
    preprocesador = ColumnTransformer(
        transformers=[
            # Aplicamos one-hot encoding a variables categóricas
            ('cat', OneHotEncoder(handle_unknown='ignore'), categoricas),
            # Aplicamos escalado estándar a variables numéricas
            ('scaler', StandardScaler(), numericas)
        ],
        remainder='passthrough'  # Mantiene otras columnas sin transformar
    )

    # Configuramos el selector de características más relevantes
    # Utiliza la prueba F para seleccionar las K mejores características
    seleccionar_k_mejores = SelectKBest(score_func=f_classif)

    # Construimos el pipeline completo con todos los pasos de procesamiento
    pipeline = Pipeline(steps=[
        # Paso 1: Preprocesamiento (one-hot encoding y escalado)
        ('preprocesador', preprocesador),
        # Paso 2: Selección de características más relevantes
        ("seleccionar_k_mejores", seleccionar_k_mejores),
        # Paso 3: Reducción de dimensionalidad con PCA
        ('pca', PCA()),
        # Paso 4: Clasificación con red neuronal multicapa
        ('clasificador', MLPClassifier(max_iter=15000, random_state=42))
    ])

    return pipeline

def hiperparametros(modelo, n_splits, x_entrenamiento, y_entrenamiento, puntuacion):
    # Configuramos GridSearchCV para búsqueda de hiperparámetros
    estimador = GridSearchCV(
        estimator=modelo,
        # Definimos la grilla de hiperparámetros a evaluar
        param_grid = {
            # PCA: usar todas las componentes principales
            'pca__n_components': [None],
            # Selección de características: usar las 20 mejores
            'seleccionar_k_mejores__k': [20],
            # Red neuronal: configuración de capas ocultas
            'clasificador__hidden_layer_sizes': [(50, 30, 40, 60)],
            # Parámetro de regularización L2
            'clasificador__alpha': [0.28],
            # Tasa de aprendizaje inicial
            'clasificador__learning_rate_init': [0.001],
        },
        cv=n_splits,        # Número de divisiones para validación cruzada
        refit=True,         # Reentrenar con mejores parámetros en todo el dataset
        scoring=puntuacion  # Métrica de evaluación a optimizar
    )
    
    # Entrenamos el modelo con búsqueda de hiperparámetros
    estimador.fit(x_entrenamiento, y_entrenamiento)

    return estimador

def metricas(modelo, x_entrenamiento, y_entrenamiento, x_prueba, y_prueba):
    # Realizamos predicciones en ambos conjuntos de datos
    y_pred_entrenamiento = modelo.predict(x_entrenamiento)  # Predicciones en entrenamiento
    y_pred_prueba = modelo.predict(x_prueba)                # Predicciones en prueba

    # Calculamos métricas para el conjunto de entrenamiento
    metricas_entrenamiento = {
        'type': 'metrics',
        'dataset': 'train',
        # Precisión: proporción de predicciones positivas correctas
        'precision': (precision_score(y_entrenamiento, y_pred_entrenamiento, average='binary')),
        # Precisión balanceada: promedio de sensibilidad y especificidad
        'balanced_accuracy':(balanced_accuracy_score(y_entrenamiento, y_pred_entrenamiento)),
        # Recall (sensibilidad): proporción de casos positivos correctamente identificados
        'recall': (recall_score(y_entrenamiento, y_pred_entrenamiento, average='binary')),
        # F1-score: media armónica entre precisión y recall
        'f1_score': (f1_score(y_entrenamiento, y_pred_entrenamiento, average='binary'))
    }

    # Calculamos métricas para el conjunto de prueba
    metricas_prueba = {
        'type': 'metrics',
        'dataset': 'test',
        # Mismas métricas calculadas para el conjunto de prueba
        'precision': (precision_score(y_prueba, y_pred_prueba, average='binary')),
        'balanced_accuracy':(balanced_accuracy_score(y_prueba, y_pred_prueba)),
        'recall': (recall_score(y_prueba, y_pred_prueba, average='binary')),
        'f1_score': (f1_score(y_prueba, y_pred_prueba, average='binary'))
    }

    return metricas_entrenamiento, metricas_prueba

def matriz_confusion(modelo, x_entrenamiento, y_entrenamiento, x_prueba, y_prueba):
    # Realizamos predicciones en ambos conjuntos de datos
    y_pred_entrenamiento = modelo.predict(x_entrenamiento)
    y_pred_prueba = modelo.predict(x_prueba)

    # Calculamos matriz de confusión para el conjunto de entrenamiento
    cm_entrenamiento = confusion_matrix(y_entrenamiento, y_pred_entrenamiento)
    # Extraemos los valores: TN, FP, FN, TP (True Negative, False Positive, False Negative, True Positive)
    tn_entrenamiento, fp_entrenamiento, fn_entrenamiento, tp_entrenamiento = cm_entrenamiento.ravel()

    # Calculamos matriz de confusión para el conjunto de prueba
    cm_prueba = confusion_matrix(y_prueba, y_pred_prueba)
    # Extraemos los valores de la matriz de confusión
    tn_prueba, fp_prueba, fn_prueba, tp_prueba = cm_prueba.ravel()

    # Estructuramos la matriz de confusión del conjunto de entrenamiento
    matriz_entrenamiento = {
        'type': 'cm_matrix',
        'dataset': 'train', 
        'true_0': {  # Casos reales negativos (no default)
            'predicted_0': int(tn_entrenamiento),  # Predichos correctamente como negativos
            'predicted_1': int(fp_entrenamiento)   # Predichos incorrectamente como positivos
        },
        'true_1': {  # Casos reales positivos (default)
            'predicted_0': int(fn_entrenamiento),  # Predichos incorrectamente como negativos
            'predicted_1': int(tp_entrenamiento)   # Predichos correctamente como positivos
        }
    }

    # Estructuramos la matriz de confusión del conjunto de prueba
    matriz_prueba = {
        'type': 'cm_matrix',
        'dataset': 'test', 
        'true_0': {  # Casos reales negativos (no default)
            'predicted_0': int(tn_prueba),    # Predichos correctamente como negativos
            'predicted_1': int(fp_prueba)     # Predichos incorrectamente como positivos
        },
        'true_1': {  # Casos reales positivos (default)
            'predicted_0': int(fn_prueba),    # Predichos incorrectamente como negativos
            'predicted_1': int(tp_prueba)     # Predichos correctamente como positivos
        }
    }

    return matriz_entrenamiento, matriz_prueba

def clean_data(modelo):
    # Creamos el directorio de destino si no existe
    os.makedirs('files/models', exist_ok=True)

    # Guardamos el modelo comprimido usando gzip y pickle
    with gzip.open('files/models/model.pkl.gz', 'wb') as f:
        pickle.dump(modelo, f)  # Serializamos el modelo con pickle

def guardar_metricas(metricas):
    # Creamos el directorio de destino si no existe
    os.makedirs('files/output', exist_ok=True)

    # Guardamos las métricas en formato JSON (una línea por diccionario)
    with open("files/output/metrics.json", "w") as f:
        for metrica in metricas:
            # Convertimos cada diccionario a JSON y lo escribimos en una línea
            linea_json = json.dumps(metrica)
            f.write(linea_json + "\n")




# Cargamos el conjunto de datos de prueba
with zipfile.ZipFile('files/input/test_data.csv.zip', 'r') as zip:
    # Abrimos el archivo CSV dentro del ZIP
    with zip.open('test_default_of_credit_card_clients.csv') as f:
        # Leemos el CSV y lo convertimos a DataFrame
        df_prueba = pd.read_csv(f)

# Cargamos el conjunto de datos de entrenamiento
with zipfile.ZipFile('files/input/train_data.csv.zip', 'r') as zip:
    # Abrimos el archivo CSV dentro del ZIP
    with zip.open('train_default_of_credit_card_clients.csv') as f:
        # Leemos el CSV y lo convertimos a DataFrame
        df_entrenamiento = pd.read_csv(f)



# Esta sección ejecuta todo el pipeline de machine learning paso a paso
if __name__ == '__main__':
    # PASO 1: LIMPIEZA DE DATOS
    # Aplicamos la función de limpieza a ambos conjuntos de datos
    df_prueba = limpiar(df_prueba)
    df_entrenamiento = limpiar(df_entrenamiento)

    # PASO 2: DIVISIÓN DE CARACTERÍSTICAS Y VARIABLE OBJETIVO
    # Separamos las características (X) de la variable objetivo (y)
    x_entrenamiento, y_entrenamiento = df_entrenamiento.drop('default', axis=1), df_entrenamiento['default']
    x_prueba, y_prueba = df_prueba.drop('default', axis=1), df_prueba['default']

    # PASO 3: CONSTRUCCIÓN DEL MODELO
    # Creamos el pipeline de machine learning
    pipeline_modelo = modelo()

    # PASO 4: OPTIMIZACIÓN DE HIPERPARÁMETROS
    # Optimizamos los hiperparámetros usando validación cruzada
    pipeline_modelo = hiperparametros(pipeline_modelo, 10, x_entrenamiento, y_entrenamiento, 'balanced_accuracy')

    # PASO 5: GUARDADO DEL MODELO
    # Guardamos el modelo entrenado y optimizado
    clean_data(pipeline_modelo)

    # PASO 6: CÁLCULO DE MÉTRICAS DE EVALUACIÓN
    # Calculamos métricas de precisión, recall, F1-score, etc.
    metricas_entrenamiento, metricas_prueba = metricas(pipeline_modelo, x_entrenamiento, y_entrenamiento, x_prueba, y_prueba)

    # PASO 7: CÁLCULO DE MATRICES DE CONFUSIÓN
    # Calculamos las matrices de confusión para análisis detallado
    matriz_entrenamiento, matriz_prueba = matriz_confusion(pipeline_modelo, x_entrenamiento, y_entrenamiento, x_prueba, y_prueba)

    # PASO 8: GUARDADO DE RESULTADOS
    # Guardamos todas las métricas y matrices de confusión
    guardar_metricas([metricas_entrenamiento, metricas_prueba, matriz_entrenamiento, matriz_prueba])
