#=====================================#
#         Logging Functions           #
#         MLops PRactice DFX5         #
#=====================================#

# libraries -------------
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

# Logger Configuration ------------
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# get arguments from command ----------
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
    exp = mlflow.set_experiment(experiment_name="experiment_custom_sklearn")
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
        "alpha": alpha,
        "l1_ratio": l1_ratio
    })

    mlflow.log_metrics({
        "rmse": rmse,
        "r2": r2,
        "mae": mae
    })

    sklearn_model_path = "sklearn_model.pkl"
    joblib.dump(lr, sklearn_model_path)
    artifacts = {
        "sklearn_model": sklearn_model_path,
        "data": data_dir
    }


    class SklearnWrapper(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            self.sklearn_model = joblib.load(context.artifacts["sklearn_model"])

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
        python_model=SklearnWrapper(),
        artifacts=artifacts,
        code_path=["main.py"],
        conda_env=conda_env
    )
    artifacts_uri = mlflow.get_artifact_uri("sklear_mlflow_pyfunc")
    mlflow.evaluate(
        artifacts_uri,
        test,
        targets="quality",
        model_type="regressor",
        evaluators=["default"]
    )

    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)
    mlflow.end_run()
    run = mlflow.last_active_run()
    print("Active run id is {}".format(run.info.run_id))
    print("Active run name is {}".format(run.info.run_name))

    # Si deseas probar con diferentes valores de alpha y l1_ratio
    for alpha in [0.1, 0.5, 0.9]:
        for l1_ratio in [0.1, 0.5, 0.9]:
            with mlflow.start_run():
                lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
                lr.fit(train_x, train_y)

                predicted_qualities = lr.predict(test_x)

                (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

                print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
                print("  RMSE: %s" % rmse)
                print("  MAE: %s" % mae)
                print("  R2: %s" % r2)

                mlflow.log_params({
                    "alpha": alpha,
                    "l1_ratio": l1_ratio
                })

                mlflow.log_metrics({
                    "rmse": rmse,
                    "r2": r2,
                    "mae": mae
                })

                sklearn_model_path = f"sklearn_model_alpha_{alpha}_l1_{l1_ratio}.pkl"
                joblib.dump(lr, sklearn_model_path)
                artifacts = {
                    "sklearn_model": sklearn_model_path,
                    "data": data_dir
                }

                mlflow.pyfunc.log_model(
                    artifact_path=f"sklear_mlflow_pyfunc_alpha_{alpha}_l1_{l1_ratio}",
                    python_model=SklearnWrapper(),
                    artifacts=artifacts,
                    code_path=["main.py"],
                    conda_env=conda_env
                )
                artifacts_uri = mlflow.get_artifact_uri(f"sklear_mlflow_pyfunc_alpha_{alpha}_l1_{l1_ratio}")
                mlflow.evaluate(
                    artifacts_uri,
                    test,
                    targets="quality",
                    model_type="regressor",
                    evaluators=["default"]
                )

                artifacts_uri = mlflow.get_artifact_uri()
                print("The artifact path is", artifacts_uri)
                mlflow.end_run()
                run = mlflow.last_active_run()
                print("Active run id is {}".format(run.info.run_id))
                print("Active run name is {}".format(run.info.run_name))
