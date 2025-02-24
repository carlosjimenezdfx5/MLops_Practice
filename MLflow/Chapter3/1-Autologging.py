#=====================================#
#         Logging Functions           #
#         MLops PRactice DFX5         #
#=====================================#


# Description --------------------

# Reference : https://mlflow.org/docs/latest/tracking/autolog.htmle
# Autologging : Is a feature that allows automatic loggign of certain parameters, metrics and artifacts

# mlflow.autolog(): autologging for each supported library that is installed
#   Params:
#       log_models: boolean field to specify whether to log the model
#       log_input_examples: Set true to log the input for training dataset like an Artifacts
#       log_model_signatures (bool): Whether to log model signatures. Defaults to True.
#       log_models (bool): Whether to log models at the end of training. Defaults to True.
#       disable (bool): Whether to disable autologging. Defaults to False.
#       exclusive (bool): Whether to disable autologging for all integrations except the current one. Defaults to False.
#       disable_for_unsupported_versions (bool): Whether to disable autologging for unsupported versions of third-party libraries. Defaults to False
#       silent (bool): Whether to suppress all autologging logs and warnings. Defaults to False.



#===============================#
# Example---------------------- #
#===============================#
# import mlflow
#
# mlflow.autolog(
#     log_input_examples=True,
#     log_model_signatures=True,
#     log_models=True,
#     disable=False,
#     exclusive=False,
#     disable_for_unsupported_versions=False,
#     silent=False
# )


#

# mlflow.<framework>.autolog(): use a specific framework for example tensorflow, autologging functions



## Steps ----------------

# 1. active server ->mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts --host 127.0.0.1 --port 5001
# 2. Into other terminal run -> python 1-tracking_server.py --path /Users/danieljimenez/Documents/projects/Labor_Projects/globant/MLflow_Mentoring/chapter1/dataset/red-wine-quality.csv
# 3.  Check http://127.0.0.1:5001


#==============================#
#      Example with code       #
#==============================#


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

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)-15s %(message)s"
)
logger = logging.getLogger()

# Design Model
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def run_experiment(args, model_name, model, model_params):
    try:
        warnings.filterwarnings("ignore")
        np.random.seed(40)

        logger.info('Loading dataset from {}'.format(args.path))
        data = pd.read_csv(args.path)

        # Create data directory
        data_dir = Path("data")
        data_dir.mkdir(parents=True, exist_ok=True)

        logger.info('Splitting dataset')
        train, test = train_test_split(data, test_size=0.25,
                                       random_state=42)
        train_x = train.drop(["quality"],
                             axis=1)
        test_x = test.drop(["quality"],
                           axis=1)
        train_y = train[["quality"]]
        test_y = test[["quality"]]

        # Save split data
        train.to_csv(data_dir / "train.csv", index=False)
        test.to_csv(data_dir / "test.csv", index=False)

        logger.info(f'Training {model_name} model')
        mlflow.set_tracking_uri('http://127.0.0.1:5001')
        mlflow.set_experiment(
            experiment_name='experiments_track_server'
        )

        with mlflow.start_run():
            mlflow.set_tag('release.version', '0.1')
            mlflow.set_tag('model_name', model_name)

            # Autologging
            mlflow.autolog(log_input_examples=True)

            # Initialize model
            model_instance = model(**model_params)
            model_instance.fit(train_x, train_y.values.ravel())

            predicted_qualities = model_instance.predict(test_x)
            rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

            # Log data directory as artifacts
            mlflow.log_artifacts(str(data_dir))

            arts = mlflow.get_artifact_uri()
            print('The uri for artifacts is {}'.format(arts))

            run = mlflow.last_active_run()
            logger.info('Active run id is {}'.format(run.info.run_id))
            logger.info('Active run name is {}'.format(run.info.run_name))

    except mlflow.exceptions.MlflowException as e:
        logger.error(f"MLflow API Error: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='design model training')
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    experiments = [
        {'model_name': 'RandomForest', 'model': RandomForestRegressor, 'model_params': {'n_estimators': 50, 'max_depth': 5}},
        {'model_name': 'RandomForest', 'model': RandomForestRegressor, 'model_params': {'n_estimators': 100, 'max_depth': 10}},
        {'model_name': 'RandomForest', 'model': RandomForestRegressor, 'model_params': {'n_estimators': 150, 'max_depth': 15}},
        {'model_name': 'RandomForest', 'model': RandomForestRegressor, 'model_params': {'n_estimators': 200, 'max_depth': 20}},
        {'model_name': 'GradientBoosting', 'model': GradientBoostingRegressor, 'model_params': {'n_estimators': 100, 'max_depth': 3}},
        {'model_name': 'SVR', 'model': SVR, 'model_params': {'C': 1.0, 'epsilon': 0.2}}
    ]

    for exp in experiments:
        run_experiment(args, exp['model_name'], exp['model'], exp['model_params'])
