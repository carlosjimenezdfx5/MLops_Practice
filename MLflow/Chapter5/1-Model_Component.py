#=====================================#
#         Logging Functions           #
#         MLops PRactice DFX5         #
#=====================================#


# MLflow models Components -----------
# Is a stardard format to packages machine learning models in a reusable
# format allowing models to be easily deployable to varios enviroments
# Reproducibility -> Collaboration_-> Flexibility

## Model Signature -----------------
# Specifies the input, output, data type and shapes the model expects and
# returns


## Model API ----------------------
# A RestAPI proving a standardized  interface for interacxting with the model


## Flavor -------------------------
# Refers to a specific way of serializing an storiung a machine learning models



# Archivo MLmodel
# El archivo MLmodel es un archivo de metadatos que se genera cuando se guarda un modelo en MLflow. Este archivo describe el modelo guardado, incluyendo la información necesaria para volver a cargarlo y ejecutarlo. Los contenidos típicos de un archivo MLmodel incluyen:
#
# Formato del modelo: Especifica el formato en el que se guarda el modelo, como python_function, sklearn, tensorflow, etc.
# Ubicación de los artefactos: Indica dónde se encuentran los archivos del modelo.
# Entradas y salidas: Describe las interfaces del modelo, como el tipo de datos de entrada y salida.
# Versiones de librerías: Puede incluir información sobre las versiones de las librerías utilizadas para entrenar el modelo.
# Custom Predict Function: Define una función de predicción personalizada si es necesaria.
# Ejemplo de un archivo MLmodel:
#
# yaml

# artifact_path: model
# flavors:
#   python_function:
#     data: model.pkl
#     env: conda.yaml
#     loader_module: mlflow.sklearn
#     python_version: 3.7.3
#   sklearn:
#     pickled_model: model.pkl
#     sklearn_version: 0.24.1
# run_id: abcdef123456
# utc_time_created: '2023-05-24T10:00:00.000000'


# El **Model Signature Enforcement** en MLflow es una característica que permite definir y validar la estructura esperada de las entradas y salidas de un modelo. Esto se logra a través de las firmas del modelo (model signatures), que especifican los tipos y formas de los datos que un modelo acepta como entrada y los datos que produce como salida.
#
# ### ¿Qué es una Model Signature?
#
# Una **model signature** es una descripción formal de los tipos de datos de entrada y salida de un modelo. Incluye:
#
# - **Input Schema**: Describe el formato y tipo de los datos que el modelo espera recibir como entrada.
# - **Output Schema**: Describe el formato y tipo de los datos que el modelo producirá como salida.
#
# Estas firmas son útiles para asegurar que los datos proporcionados al modelo durante la inferencia coincidan con los datos con los que fue entrenado, evitando errores de incompatibilidad.
#
# ### Ejemplo de una Model Signature
#
# Aquí tienes un ejemplo de cómo se ve una model signature en MLflow:
#
# ```python
# from mlflow.models.signature import infer_signature
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
#
# # Entrenar un modelo de ejemplo
# data = pd.DataFrame({
#     "feature1": [1.0, 2.0, 3.0, 4.0],
#     "feature2": [10.0, 20.0, 30.0, 40.0]
# })
# target = [0, 1, 0, 1]
# model = LogisticRegression().fit(data, target)
#
# # Inferir la firma del modelo
# signature = infer_signature(data, model.predict(data))
#
# # Guardar el modelo con la firma en MLflow
# import mlflow
# mlflow.sklearn.log_model(model, "model", signature=signature)
# ```
#
# ### ¿Qué es el Model Signature Enforcement?
#
# El **Model Signature Enforcement** se refiere a la validación que ocurre cuando se carga un modelo guardado en MLflow y se proporciona una entrada para la inferencia. MLflow valida que los datos de entrada coincidan con la firma del modelo guardado. Si los datos no coinciden, se genera un error, lo que ayuda a garantizar que el modelo no se ejecute con datos no válidos.
#
# ### Beneficios del Model Signature Enforcement
#
# 1. **Consistencia**: Asegura que los datos proporcionados para la inferencia coincidan con el formato y tipo de datos utilizados durante el entrenamiento.
# 2. **Reducción de Errores**: Previene errores y problemas de compatibilidad que pueden surgir cuando los datos de entrada no coinciden con las expectativas del modelo.
# 3. **Reproducibilidad**: Facilita la reproducibilidad al documentar explícitamente la estructura esperada de los datos.
#
# ### Cómo Funciona
#
# Cuando un modelo se guarda con una firma en MLflow, se incluye la firma en el archivo `MLmodel`. Durante la inferencia, MLflow compara los datos de entrada proporcionados con la firma guardada:
#
# - Si los datos coinciden con la firma, la inferencia procede.
# - Si los datos no coinciden, se genera un error de validación.
#
# ### Ejemplo de Validación de Firmas
#
# ```python
# import mlflow.pyfunc
#
# # Cargar el modelo guardado
# model = mlflow.pyfunc.load_model("path/to/saved/model")
#
# # Proveer datos de entrada para la inferencia
# input_data = pd.DataFrame({
#     "feature1": [5.0, 6.0],
#     "feature2": [50.0, 60.0]
# })
#
# # Realizar la inferencia (esto validará los datos de entrada contra la firma)
# predictions = model.predict(input_data)
# ```
#
# Si `input_data` no coincide con la firma del modelo, MLflow lanzará una excepción indicando que los datos de entrada no son válidos.
#
# En resumen, el **Model Signature Enforcement** es una característica poderosa de MLflow que ayuda a mantener la consistencia y la calidad de los datos a lo largo del ciclo de vida del modelo, desde el entrenamiento hasta la inferencia.

#==============================#
#      Example with code       #
#==============================#



## Design Model ----------
# libraries -------------
# libraries -------------
import warnings
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import mlflow
import mlflow.sklearn
from pathlib import Path
from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.types.schema import Schema, ColSpec

# Configure logger --------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)-15s %(message)s"
)
logger = logging.getLogger()


## Design Model ----------
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def run_experiment(args, model_name, model, model_params):
    """
    Load data, train model, and evaluate model.

    :param args: Namespace, command line arguments
    :param model_name: Name of the model to be used
    :param model: The model class
    :param model_params: Dictionary of parameters for the model
    """
    try:
        warnings.filterwarnings("ignore")
        np.random.seed(40)

        logger.info('Loading dataset from {}'.format(args.path))
        data = pd.read_csv(args.path)

        # Create data directory
        data_dir = Path("data")
        data_dir.mkdir(parents=True, exist_ok=True)

        logger.info('Splitting dataset')
        train, test = train_test_split(data, test_size=0.25, random_state=42)
        train_x = train.drop(["quality"], axis=1)
        test_x = test.drop(["quality"], axis=1)
        train_y = train[["quality"]]
        test_y = test[["quality"]]

        # Save split data
        train.to_csv(data_dir / "train.csv", index=False)
        test.to_csv(data_dir / "test.csv", index=False)

        logger.info(f'Training {model_name} model')
        mlflow.set_tracking_uri('')
        mlflow.set_experiment(experiment_name='Components_experiments')

        with mlflow.start_run():
            mlflow.set_tag('release.version', '0.1')
            mlflow.set_tag('model_name', model_name)

            # Autologging ----------
            mlflow.sklearn.autolog()

            # Initialize model ------
            model_instance = model(**model_params)
            model_instance.fit(train_x, train_y.values.ravel())

            predicted_qualities = model_instance.predict(test_x)
            rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

            # Log data directory as artifacts ----------
            mlflow.log_artifacts(str(data_dir))

            # Logging the model with signature and input example
            input_data = [
                {"name": "fixed acidity", "type": "double"},
                {"name": "volatile acidity", "type": "double"},
                {"name": "citric acid", "type": "double"},
                {"name": "residual sugar", "type": "double"},
                {"name": "chlorides", "type": "double"},
                {"name": "free sulfur dioxide", "type": "double"},
                {"name": "total sulfur dioxide", "type": "double"},
                {"name": "density", "type": "double"},
                {"name": "pH", "type": "double"},
                {"name": "sulphates", "type": "double"},
                {"name": "alcohol", "type": "double"},
                {"name": "quality", "type": "double"}
            ]

            output_data = [{'type': 'double'}]  # Changed 'long' to 'double' to match quality type

            input_schema = Schema([ColSpec(col["type"], col['name']) for col in input_data])
            output_schema = Schema([ColSpec(col['type']) for col in output_data])

            signature = ModelSignature(inputs=input_schema, outputs=output_schema)

            input_example = {
                "fixed acidity": 7.2,
                "volatile acidity": 0.35,
                "citric acid": 0.45,
                "residual sugar": 8.5,
                "chlorides": 0.045,
                "free sulfur dioxide": 30.0,
                "total sulfur dioxide": 120.0,
                "density": 0.997,
                "pH": 3.2,
                "sulphates": 0.65,
                "alcohol": 9.2,
                "quality": 6.0
            }

            mlflow.sklearn.log_model(model_instance, "model", signature=signature, input_example=input_example)

            run = mlflow.last_active_run()
            logger.info('Active run id is {}'.format(run.info.run_id))
            logger.info('Active run name is {}'.format(run.info.run_name))

    except Exception as e:
        logger.error(f"An error occurred: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='design model training')
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    experiments = [
        {'model_name': 'RandomForest', 'model': RandomForestRegressor,
         'model_params': {'n_estimators': 50, 'max_depth': 5}},
        {'model_name': 'RandomForest', 'model': RandomForestRegressor,
         'model_params': {'n_estimators': 100, 'max_depth': 10}},
        {'model_name': 'RandomForest', 'model': RandomForestRegressor,
         'model_params': {'n_estimators': 150, 'max_depth': 15}},
        {'model_name': 'RandomForest', 'model': RandomForestRegressor,
         'model_params': {'n_estimators': 200, 'max_depth': 20}},
        {'model_name': 'GradientBoosting', 'model': GradientBoostingRegressor,
         'model_params': {'n_estimators': 100, 'max_depth': 3}},
        {'model_name': 'SVR', 'model': SVR, 'model_params': {'C': 1.0, 'epsilon': 0.2}}
    ]

    for exp in experiments:
        run_experiment(args, exp['model_name'], exp['model'], exp['model_params'])
