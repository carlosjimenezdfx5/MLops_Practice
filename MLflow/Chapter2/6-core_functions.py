#=====================================#
#         Logging Functions           #
#         MLops PRactice DFX5         #
#=====================================#


# Reference ---------
# https://mlflow.org/docs/latest/python_api/mlflow.html

# Functions ---------
###############
#    Params   #
###############

# Log(s) params: Used to log the model- hyperparameters

# mlflow.log_param(): single hyperpameter as key-value pair
#   params:
#       keys(str) -> Name of the parameter to log
#       value(Any) ->  Values to register into the experiment




# mlflow.log_params(): Multiples hyperpametesr as keys-values pairs
#   params:
#       params(Dict)[str,Any] -> Dictionary of param_name Relation string -value

###############
#  metrics    #
###############

# Used to log the Model - Metrics

# mlflow.log_metric(): Logs a single metric as Key-value pair
#   params:
#       Key -> Name of the metric to log
#       Value -> value of the metric
#       step -> A single integer step at wich to log the specified metrics (deep learning)





# mlflow.log_metrics() : Logs a multiple metrics as Key-value pairs
#   params:
#       Metrics -> Dictionary of metrics_names Relation String-Value
#       Step -> A single integer step at witch to log the specified Metrics





###############
#  Artifacts  #
###############
# Used to log the Model - Artifact

# mlflow.artifact() : Log a single artifact
#   params:
#       local_path:str -> Path of the file that to be stored
#       artifact_path:str -> Path to store the artifacts





# mlflow.log_artifacts() : Log a multiple artifacts
#   params:
#       local_path:str -> Path of the file that to be stored
#       artifact_path:str -> Path to store the artifacts



#mlflow.get_artifact_uri(): get absolute URL of the specified artifact
#   Params:
#       Artifact_path: The run-relative artifact


## Example with code -----------



import warnings
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn
from pathlib import Path


# Configure logger --------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)-15s %(message)s"
)
logger = logging.getLogger(__name__)


## Design Model ----------

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def run_model(args):
    """
    Load data, train model, and evaluate model.

    :param args: Namespace, command line arguments
    """
    try:
        warnings.filterwarnings("ignore")
        np.random.seed(40)

        logger.info('Loading dataset from {}'.format(args.path))
        data = pd.read_csv(args.path)

        # Create data directory
        Path("data").mkdir(parents=True, exist_ok=True)

        logger.info('Splitting dataset')
        train, test = train_test_split(data, test_size=0.25, random_state=42)
        train_x = train.drop(["quality"], axis=1)
        test_x = test.drop(["quality"], axis=1)
        train_y = train[["quality"]]
        test_y = test[["quality"]]

        # Save split data
        train.to_csv("data/train.csv", index=False)
        test.to_csv("data/test.csv", index=False)

        logger.info('Training RandomForest model')
        #==========================
        # Here  using MLflow      #
        #==========================
        mlflow.set_tracking_uri('')
        # Here change to set_experiment -------
        get_id = mlflow.set_experiment(
            experiment_name='experiments_1') # Change this name

        mlflow.start_run()
        mlflow.set_tag('release.version','0.1')
        rf = RandomForestRegressor(n_estimators=args.n_estimators,
                                       max_depth=args.max_depth,
                                       random_state=42)
        rf.fit(train_x, train_y.values.ravel())

        predicted_qualities = rf.predict(test_x)
        rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

        #================================================#
        # Register multiple Logs params ---------------- #
        #================================================#
        logger.info(f"RandomForest model (n_estimators={args.n_estimators}, max_depth={args.max_depth}):")
        # Log Params ---------
        params = {
            'n_estimators': args.n_estimators,
            'max_depth':args.max_depth
        }
        mlflow.log_params(params)

        # Log Metrics ----------
        metrics = {
            'rmse' : rmse,
            'r2':r2
        }

        mlflow.log_metrics(metrics)

        # Log Artifact ----------
        mlflow.sklearn.log_model(rf, 'first_new_model')

        # Log data directory as artifacts ----------
        mlflow.log_artifacts('data')

        arts = mlflow.get_artifact_uri()
        print('The uri for artifacts is {}'.format(arts))

        run = mlflow.last_active_run()
        logger.info('Active run id is {}'.format(run.info.run_id))
        logger.info('Active run name is {}'.format(run.info.run_name))
        # MLflow end run --------
        mlflow.end_run()

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='design model training'
    )
    parser.add_argument("--path",
                        type=str,
                        required=True)
    parser.add_argument("--n_estimators",
                        type=int,
                        required=True,
                        default=50)
    parser.add_argument("--max_depth",
                        type=int,
                        required=True,
                        default=5)
    args = parser.parse_args()
    run_model(args)


#python 6-core_functions.py --path /Users/danieljimenez/Documents/projects/Labor_Projects/globant/MLflow_Mentoring/chapter1/dataset/red-wine-quality.csv --n_estimators 15 --max_depth 30

