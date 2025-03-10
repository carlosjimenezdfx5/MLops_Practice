#=====================================#
#         Logging Functions           #
#        MLops PRactice DFX5          #
#=====================================#


# Descriptions ===========================
# Experiment Tracking: With logging functions, you can keep a record of different experiments you conduct,
# including the parameters used, the performance metrics, and any additional notes or tags. This makes it easier to compare
# different models and understand what works best for your task.

# Model Versioning: Logging functions allow you to log versions of your models along with their performance metrics.
# This helps in keeping track of model iterations and facilitates reproducibility by ensuring that you can always go back to previous versions if needed.

# Performance Monitoring: You can use logging functions to monitor the performance of your models over time.
# This includes tracking metrics like accuracy, loss, and other custom metrics specific to your model and task.
# This monitoring is crucial for model maintenance and optimization.

#Visualization and Reporting: Logging functions can also be used to log artifacts such as plots, images, and model files.
# This enables you to visualize your model's performance and results, create reports, and share insights with collaborators or stakeholders.



# Main Functions ====================
# set_tracking_uri() : Set the tracking URI of choice
#  Params :
#       -> URI : Location where files
#             i) Empty String : MLflow create a floders with mlruis
#             ii) Folder Name
#             iii) File Path
# get_traking_uri() :  Get set tracking URI


# Run: mlflow ui --backend-store-uri ./tracks_experiments/

## Example #2 ===================


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
        mlflow.set_tracking_uri('./alex_experiments/')
        logger.info('The set tracking uri is {}'.format(mlflow.get_tracking_uri()))
        exp = mlflow.set_experiment(
            experiment_name= 'exp_1_rf_uri'
        )
        with mlflow.start_run(experiment_id=exp.experiment_id):
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