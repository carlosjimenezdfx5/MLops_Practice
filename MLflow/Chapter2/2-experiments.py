#=====================================#
#         Logging Functions           #
#        MLops PRactice DFX5          #
#=====================================#


# Descriptions =========

# create_experiments() :
#   Params:
#       -> Name
#       -> artifact_location
#       -> tags
# set_experiments() : set already created experiment

## Example #2 with experiment tracks




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

        logger.info('Splitting dataset')
        train, test = train_test_split(data, test_size=0.25, random_state=42)
        train_x = train.drop(["quality"], axis=1)
        test_x = test.drop(["quality"], axis=1)
        train_y = train[["quality"]]
        test_y = test[["quality"]]

        logger.info('Training RandomForest model')
        #==========================
        # Here  using MLflow      #
        #==========================
        mlflow.set_tracking_uri('')
        # Here change to create_experiment -------
        exp_id = mlflow.create_experiment(
            name='RanForest_Example_v2',
            tags={'version': 'v2',
                  'priority':'p2'},
            artifact_location=Path.cwd().joinpath('experiments').as_uri()# Create new folder
        )

        # Get experiment ---------------
        get_id = mlflow.get_experiment(exp_id)

        print('Name of Experiment: {}'.format(get_id.name))
        print('Lifecycle_stage : {}'.format(get_id.lifecycle_stage))

        with mlflow.start_run(experiment_id=get_id.experiment_id):
            rf = RandomForestRegressor(n_estimators=args.n_estimators,
                                       max_depth=args.max_depth,
                                       random_state=42)
            rf.fit(train_x, train_y.values.ravel())

            predicted_qualities = rf.predict(test_x)
            rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

            logger.info(f"RandomForest model (n_estimators={args.n_estimators}, max_depth={args.max_depth}):")
            logger.info(f"RMSE: {rmse}")
            logger.info(f"MAE: {mae}")
            logger.info(f"R2: {r2}")
            mlflow.log_param('n_estimators', args.n_estimators)
            mlflow.log_param('max_depth', args.max_depth)
            mlflow.log_metric('rmse', rmse)
            mlflow.log_metric('r2', r2)
            mlflow.sklearn.log_model(rf,'first_model')


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