#=================================================#
#                 MLflow Introduction             #
#                DFX5 MLOPS Practice              #
#=================================================#

"""
MLflow Time Series Forecasting with AWS and TensorFlow
======================================================
This script demonstrates how to use MLflow to track experiments, register models,
and perform automatic logging using AWS as an artifact store. It implements a deep
learning model using LSTM with fine-tuning via Keras-Tuner for hyperparameter optimization.

Key Features:
- MLflow integration for experiment tracking and model registry.
- AWS S3-based artifact storage.
- LSTM model for time series forecasting.
- Hyperparameter tuning using Keras-Tuner.
- Automatic logging of training parameters and metrics.
"""
# libraries ----------
import mlflow
import mlflow.tensorflow
import tensorflow as tf
import keras_tuner as kt
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from mlflow.models.signature import infer_signature

# Configure MLflow with AWS S3 as the artifact store
mlflow.set_tracking_uri("http://your-mlflow-server:5000")  # Replace with your MLflow server URL
mlflow.set_experiment("AWS_MLflow_TimeSeries")

# Load sample time series data (synthetic or real dataset)
np.random.seed(42)
data = np.sin(np.linspace(0, 100, 1000)) + np.random.normal(scale=0.1, size=1000)
data = data.reshape(-1, 1)

# Normalize data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Prepare training data
def create_dataset(data, look_back=20):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i : i + look_back])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

look_back = 20
X, y = create_dataset(data_scaled, look_back)
X_train, y_train = X[:800], y[:800]
X_test, y_test = X[800:], y[800:]

# Define an LSTM model with Keras-Tuner hyperparameter tuning
def model_builder(hp):
    model = Sequential()
    model.add(LSTM(hp.Int('units_1', min_value=32, max_value=128, step=32), return_sequences=True, input_shape=(look_back, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(hp.Int('units_2', min_value=32, max_value=128, step=32)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Choice('lr', [1e-2, 1e-3, 1e-4])),
                  loss='mse')
    return model

# Enable auto logging --------
mlflow.tensorflow.autolog()

# Hyperparameter tuning using Keras-Tuner
tuner = kt.Hyperband(model_builder,
                     objective='val_loss',
                     max_epochs=20,
                     factor=3,
                     directory='kt_logs',
                     project_name='time_series_tuning')

with mlflow.start_run() as run:
    tuner.search(X_train, y_train, epochs=50, validation_data=(X_test, y_test))
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    # Train the best model
    best_model = tuner.hypermodel.build(best_hps)
    history = best_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=16, verbose=1)
    
    # Save model signature
    signature = infer_signature(X_test, best_model.predict(X_test))
    
    # Log model
    mlflow.tensorflow.log_model(best_model, "lstm_model", signature=signature)
    
    print(f"Run ID: {run.info.run_id}")

print("Experiment tracking complete. Model registered in MLflow.")
