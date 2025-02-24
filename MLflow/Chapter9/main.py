'''
MLproject
'''

import os
import warnings
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
from pathlib import Path
import mlflow
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = np.mean(np.abs(actual - pred))
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, required=False, default=0.4)
    parser.add_argument("--l1_ratio", type=float, required=False, default=0.4)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file
    data = pd.read_csv("/Users/danieljimenez/Documents/Projects/Labor_Projects/globant/MLflow_Mentoring/chapter1/dataset/red-wine-quality.csv")
    data.to_csv("data/red-wine-quality.csv", index=False)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = args.alpha
    l1_ratio = args.l1_ratio

    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    experiment = mlflow.set_experiment(experiment_name="Project exp 1")

    print("Name: {}".format(experiment.name))
    print("Experiment_id: {}".format(experiment.experiment_id))

    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        lr = ElasticNet(alpha=args.alpha, l1_ratio=args.l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(args.alpha, args.l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_metrics({
            "rmse": rmse,
            "r2": r2,
            "mae": mae
        })

        mlflow.log_params({
            "alpha": args.alpha,
            "l1_ratio": args.l1_ratio
        })

        mlflow.sklearn.log_model(lr, "model")

        run_id = run.info.run_id
        print(f"Run ID: {run_id}")

if __name__ == "__main__":
    main()


# set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
# mlflow run --entry-point ElasticNet -P alpha=0.5