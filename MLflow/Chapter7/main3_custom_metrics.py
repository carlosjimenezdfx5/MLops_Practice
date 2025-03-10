#=====================================#
#         Logging Functions           #
#         MLops PRactice DFX5         #
#=====================================#

# Model Evaluate --------
# mlflow.evaluate() : Api porvided MLflow to evaluate thge perforrmance of MLflow models
# and saves evaluations metrics and graphs to the tracking server

# evaluate(): API evaluate the performance of MLflow Models
# Params :
#   model -> represents the model to be evaluated
#   data -> Specifies the evaluation data
# model_type -> describes the type of the model . Support 'Regressor' and classifier, Q-A, Text and text-summarization
# targets -> Lis of evaluation labels
# dataset_path -> OPtional parameter representing the pat where data s stored
# feature_names -> names of the features
# evaluators -> specify a list of evaluator names used for model evaluatrion mlflow.models.list_evaluators()
# evaluator_config -> addtioniontla configuration
# log_model_expklainability
# explainability
# custom_metrics
# custom_artifacts
# validation_thresholds -> specifies custom thresholds for classification metrics like precision, recall and f1score


## Code ------------

import warnings
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
import mlflow.sklearn
from pathlib import Path
import os
from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.types.schema import Schema, ColSpec
import sklearn
import joblib
import cloudpickle
import matplotlib.pyplot as plt
from mlflow.models import make_metric
from sklearn.dummy import DummyRegressor
from mlflow.models import MetricThreshold

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# get arguments from command
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=False, default=0.7)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.7)
args = parser.parse_args()


# evaluation function
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    data = pd.read_csv(
        "/Users/danieljimenez/Documents/Projects/Labor_Projects/globant/MLflow_Mentoring/chapter1/dataset/red-wine-quality.csv")
    # os.mkdir("data/")
    data.to_csv("data/red-wine-quality.csv", index=False)
    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    data_dir = 'red-wine-data'

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    data.to_csv(data_dir + '/data.csv')
    train.to_csv(data_dir + '/train.csv')
    test.to_csv(data_dir + '/test.csv')

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = args.alpha
    l1_ratio = args.l1_ratio

    mlflow.set_tracking_uri(uri="")

    print("The set tracking uri is ", mlflow.get_tracking_uri())
    exp = mlflow.set_experiment(experiment_name="experiment_model_evaluation")
    # get_exp = mlflow.get_experiment(exp_id)

    print("Name: {}".format(exp.name))
    print("Experiment_id: {}".format(exp.experiment_id))
    print("Artifact Location: {}".format(exp.artifact_location))
    print("Tags: {}".format(exp.tags))
    print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
    print("Creation timestamp: {}".format(exp.creation_time))

    mlflow.start_run()
    tags = {
        "engineering": "ML platform",
        "release.candidate": "RC1",
        "release.version": "2.0"
    }

    mlflow.set_tags(tags)
    mlflow.sklearn.autolog(
        log_input_examples=False,
        log_model_signatures=False,
        log_models=False
    )
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    mlflow.log_params({
        "alpha": 0.7,
        "l1_ratio": 0.7
    })

    mlflow.log_metrics({
        "rmse": rmse,
        "r2": r2,
        "mae": mae
    })

    baseline_model = DummyRegressor()
    baseline_model.fit(train_x, train_y)

    baseline_predicted_qualites = baseline_model.predict(test_x)

    bl_rmse, bl_mae, bl_r2 = eval_metrics(test_y, baseline_predicted_qualites)

    print("Baseline Dummy model")
    print("  Baseline RMSE: %s" % bl_rmse)
    print("  Baseline MAE: %s" % bl_mae)
    print("  Baseline R2: %s" % bl_r2)

    mlflow.log_metrics({
        "Baseline rmse": bl_rmse,
        "Baseline r2": bl_rmse,
        "Baseline mae": bl_rmse
    })

    sklearn_model_path = "sklearn_model.pkl"
    joblib.dump(lr, sklearn_model_path)
    artifacts = {
        "sklearn_model": sklearn_model_path,
        "data": data_dir
    }

    baseline_sklearn_model_path = "baseline_sklearn_model.pkl"
    joblib.dump(baseline_model, baseline_sklearn_model_path)
    baseline_artifacts = {"baseline_sklearn_model": baseline_sklearn_model_path}


    class SklearnWrapper(mlflow.pyfunc.PythonModel):

        def __init__(self, artifacts_name):
            self.artifacts_name = artifacts_name

        def load_context(self, context):
            self.sklearn_model = joblib.load(context.artifacts[self.artifacts_name])

        def predict(self, context, model_input):
            return self.sklearn_model.predict(model_input.values)


    # Create a Conda environment for the new MLflow Model that contains all necessary dependencies.
    conda_env = {
        "channels": ["defaults"],
        "dependencies": [
            "python={}".format(3.10),
            "pip",
            {
                "pip": [
                    "mlflow=={}".format(mlflow.__version__),
                    "scikit-learn=={}".format(sklearn.__version__),
                    "cloudpickle=={}".format(cloudpickle.__version__),
                ],
            },
        ],
        "name": "sklearn_env",
    }

    mlflow.pyfunc.log_model(
        artifact_path="sklear_mlflow_pyfunc",
        python_model=SklearnWrapper("sklearn_model"),
        artifacts=artifacts,
        code_path=["main.py"],
        conda_env=conda_env
    )

    mlflow.pyfunc.log_model(
        artifact_path="baseline_sklearn_mlflow_pyfunc",
        python_model=SklearnWrapper("baseline_sklearn_model"),
        artifacts=baseline_artifacts,
        code_path=["main.py"],
        conda_env=conda_env
    )


    def squared_diff_plus_one(eval_df, _builtin_metrics):
        return np.sum(np.abs(eval_df["prediction"] - eval_df["target"] + 1) ** 2)


    def sum_on_target_divided_by_two(_eval_df, builtin_metrics):
        return builtin_metrics["sum_on_target"] / 2


    squared_diff_plus_one_metric = make_metric(
        eval_fn=squared_diff_plus_one,
        greater_is_better=False,
        name="squared diff plus one"
    )

    sum_on_target_divided_by_two_metric = make_metric(
        eval_fn=sum_on_target_divided_by_two,
        greater_is_better=True,
        name="sum on target divided by two"
    )


    def prediction_target_scatter(eval_df, _builtin_metrics, artifacts_dir):
        plt.scatter(eval_df["prediction"], eval_df["target"])
        plt.xlabel("Targets")
        plt.ylabel("Predictions")
        plt.title("Targets vs. Predictions")
        plot_path = os.path.join(artifacts_dir, "example_scatter_plot.png")
        plt.savefig(plot_path)
        return {"example_scatter_plot_artifact": plot_path}


    artifacts_uri = mlflow.get_artifact_uri("sklear_mlflow_pyfunc")

    thresholds = {
        "mean_squared_error": MetricThreshold(
            threshold=0.6,  # Maximum MSE threshold
            min_absolute_change=0.1,  # Minimum absolute improvement compared to baseline
            min_relative_change=0.05,  # Minimum relative improvement compared to baseline
            greater_is_better=False  # Lower MSE is better
        )
    }
    baseline_model_uri = mlflow.get_artifact_uri("baseline_sklearn_mlflow_pyfunc")
    mlflow.evaluate(
        artifacts_uri,
        test,
        targets="quality",
        model_type="regressor",
        evaluators=["default"],
        custom_metrics=[
            squared_diff_plus_one_metric,
            sum_on_target_divided_by_two_metric
        ],
        custom_artifacts=[prediction_target_scatter],
        validation_thresholds=thresholds,
        baseline_model=baseline_model_uri
    )

    print("The artifact path is", artifacts_uri)
    mlflow.end_run()
    run = mlflow.last_active_run()
    print("Active run id is {}".format(run.info.run_id))
    print("Active run name is {}".format(run.info.run_name))