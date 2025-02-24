#=================================================#
#                 MLflow Introduction             #
#                 MLOPS Practice                  #
#=================================================#


# Why Using MLflow ? =============
# Is fundamental to MLOPs
# i) A set principal practices to ML lifecycle management
# ii) Monitoring the machine learning pipelines
# iii) CI/CD

# Traditional MLcycle ==================
# 1) Business Understanding
# 2) Data Acquisition
#   2.1) Identify Formats
#   2.2) Identify data sources
#   2.3) DataLake Creations
#   2.4) Design Conections
# 3) DataOps Process
#   3.1) EDA
#   3.2) Data Transformation
#   3.3) Data Preprocessing
# 4) Model design
#   4.1) Select Algos
#   4.2) Model performance
#   4.3) Hyperparams tunning
# 5) Model Deployment
# 6) Monitoring

# Developments and Operations is attending into separate groups
# DS :
#   -> Data Acquisition, Data Processing, model Building, Training Model , model Validation
# Ops:
#   -> Package, compile , Deploy, Release, and Monitoring


# Productionize a Model ==================
# i) Build and test locally
# ii) Package:
#    -> Compile the code
#    -> Resolve dependencies
#    -> Run Scripts
# iii) Performance
#    -> Scaling (inference)
#    -> Model performance
#    -> Load Balancing
#    -> Add Parallelms (Dask)
#    -> Predictions Speed
# iv) Versioning and monitoring
# v) Automatization
#   -> Continuos Training


# Principles ===========================
# i) Transition Friction: Design algos like a Microservices
#     Microservices : (Model_{design} F_model_tracking(mlflow))+(Docker/conda.yml + FastAPI)
# ii) Version Control System
# iii) Performance -> Distributed computing and containerization in Docker or Kubernetes
# iv) Automatization -> Build Workflows and CI/CD
# v) Monitoring
# vi) Continuous Training -> Tesla Enginee


# MLflow ===================
# https://mlflow.org/

# MLflow is a platform that helps manage the machine learning lifecycle,
# including experimentation, reproducibility, and deployment. It provides tools to track experiments,
# package code into reproducible runs, and share and deploy machine learning models.

# Components
# i) Tracking : Track experiment, compare parameters and results
# ii) Projects : Code to ensure reusability and reproducibility
# iii) Models : Packing models
# iv) Registry : Central model store


# Installing ===================

# ```{terminal}
# pip install mlflow
# ```


# First Example =================

# Run : python3 class\#1/class1.py --path /Users/danieljimenez/Desktop/MLflow_Mentoring/chapter1/dataset/red-wine-quality.csv --n_estimators 100 --max_depth 2

# Libraries -------------
import warnings
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor



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
        train, test = train_test_split(data,
                                       test_size=0.25,
                                       random_state=42)
        train_x = train.drop(["quality"], axis=1)
        test_x = test.drop(["quality"], axis=1)
        train_y = train[["quality"]]
        test_y = test[["quality"]]

        logger.info('Training RandomForest model')
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

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__=='__main__':
    # First Step : Define parser
    parser = argparse.ArgumentParser(
        description='design model training'
    )
    # Adding the parser (Conditions)
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
    # 3rd set the parse
    args = parser.parse_args()
    run_model(args)



