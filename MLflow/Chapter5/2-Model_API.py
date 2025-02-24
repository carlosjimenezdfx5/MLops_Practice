#=====================================#
#         Logging Functions           #
#         MLops PRactice DFX5         #
#=====================================#



# Description --------------
# save_model() : Save model to path on local file system
#   Params:
#       sk_model  -> Scikit-learn model to be saved
#       path -> Local_file where the model is to saved
#       conda_env -> Conda.env
#       code_paths -> List of local filesystem path to python dependencies
#       mlflow_model -> flavor
#       serialization_format: format to serialize the model
#       Signature -> MOdel Signature class
#       input_example
#       pip_requirements -> pip requirements
#       pyfunc_predict_fn -> Nam of the prediction function
#       metadata-> Custom metadata dictionary


# log_model() los model like an Artifact
#   Params:
#       artifact_path
#       register_model_name -> register a model with the specificied name
#       await_registration_for -> # seconds to save model


# load_model():
#   Params:
#       dst_path: Local fileSystem path to which to download artifact
#       model_uri -> location of MLflow model



#En MLflow, un flavor es una forma estándar de describir cómo se debe cargar y servir un modelo. Los flavors permiten que los modelos guardados en MLflow sean reutilizables y portables entre diferentes frameworks y entornos. Cada flavor define una interfaz específica para interactuar con el modelo, facilitando su uso y despliegue en diversas aplicaciones.

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

            #mlflow.sklearn.log_model(model_instance, "model", signature=signature, input_example=input_example)
            mlflow.sklearn.save_model(model_instance, "model", signature=signature, input_example=input_example)

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
