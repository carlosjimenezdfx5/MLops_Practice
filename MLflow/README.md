# MLflow DFX5 Practice Documentation

## Overview
This repository serves as a comprehensive guide to **MLflow**, an open-source platform for managing the ML lifecycle. The repository is structured to help users understand and implement MLflow functionalities in a progressive manner. Each chapter represents a standalone module focusing on different aspects of MLflow, from experiment tracking to model deployment and advanced ML operations.

## Repository Structure

```
MLflow/
│── conda.yml
│── README.md
│── IMG/
│── chapter1/
│   ├── introduction.md
│   ├── tracking_basics.py
│   ├── experiment_setup.py
│── chapter2/
│   ├── logging_parameters.py
│   ├── logging_metrics.py
│── Chapter3/
│   ├── artifact_management.py
│   ├── storing_data.md
│── Chapter4/
│   ├── model_registry.md
│   ├── registering_models.py
│── Chapter5/
│   ├── serving_models.md
│   ├── deployment.py
│── Chapter6/
│   ├── advanced_experiment_tracking.md
│   ├── custom_logging.py
│── Chapter7/
│   ├── hyperparameter_tuning.md
│   ├── bayesian_optimization.py
│── Chapter8/
│   ├── predictions.py
│   ├── conda.yml
│   ├── list_models.py
│   ├── assign_alias.py
│   ├── main.py
│── Chapter9/
│   ├── scaling_mlflow.md
│   ├── distributed_tracking.py
│── Chapter10/
│   ├── Readme.md
│   ├── 1-main.py
│── Chapter11/
│   ├── Readme.md
│   ├── main.py

```

### **1. Root Files**
- **`conda.yml`**: Defines dependencies for setting up the MLflow environment. Install with:
  ```bash
  conda env create -f conda.yml
  ```
- **`README.md`**: This documentation file, describing the project structure and purpose.


### **2. Chapter Directories**
Each chapter is dedicated to a specific aspect of MLflow:

#### **`chapter1/` - Getting Started with MLflow**
- **`introduction.md`**: Overview of MLflow and its core components.
- **`tracking_basics.py`**: Basic MLflow tracking examples.
- **`experiment_setup.py`**: Setting up a structured experiment.

#### **`chapter2/` - Logging Experiment Details**
- **`logging_parameters.py`**: How to log hyperparameters with MLflow.
- **`logging_metrics.py`**: Logging and visualizing performance metrics.

#### **`Chapter3/` - Managing Artifacts**
- **`artifact_management.py`**: Storing and retrieving artifacts in MLflow.
- **`storing_data.md`**: Best practices for handling ML artifacts.

#### **`Chapter4/` - Model Registry**
- **`model_registry.md`**: Guide to versioning and organizing models.
- **`registering_models.py`**: Code to register models programmatically.

#### **`Chapter5/` - Model Deployment**
- **`serving_models.md`**: MLflow model serving strategies.
- **`deployment.py`**: Deploying an MLflow model using a REST API.

#### **`Chapter6/` - Advanced Experiment Tracking**
- **`advanced_experiment_tracking.md`**: Customizing MLflow tracking.
- **`custom_logging.py`**: Implementing advanced logging mechanisms.

#### **`Chapter7/` - Hyperparameter Optimization**
- **`hyperparameter_tuning.md`**: Strategies for tuning hyperparameters.
- **`bayesian_optimization.py`**: Implementing Bayesian Optimization with MLflow.

#### **`Chapter8/` - Model Management & Predictions**
- **`predictions.py`**: Running model inference.
- **`list_models.py`**: Listing registered models.
- **`assign_alias.py`**: Assigning aliases to MLflow models.
- **`main.py`**: Main workflow script handling model interactions.
- **`conda.yml`**: Environment dependencies for this chapter.

#### **`Chapter9/` - Scaling MLflow**
- **`scaling_mlflow.md`**: Best practices for scaling MLflow.
- **`distributed_tracking.py`**: Implementing distributed experiment tracking.

#### **`Chapter10/` - Scaling MLflow**
- **`1-main`**: Best practices for scaling MLflow on AWS.
- **`Readme`**: Manual

#### **`Chapter11/` - Scaling MLflow**
- **`1-main`**: LLMops Schema
- **`Readme`**: Manual

### **3. IMG/**
Contains visual aids used in documentation, including model lifecycle diagrams and MLflow UI snapshots.

## Installation & Setup
1. Clone the repository:
   
2. Create and activate the Conda environment:
   ```bash
   conda env create -f conda.yml
   conda activate dfx5-mlflow
   ```
3. Start the MLflow tracking server:
   ```bash
   mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
   ```

## Running the Code
Each chapter contains scripts that can be run independently.
Example: Running model inference from Chapter 8:
```bash
cd Chapter8
python predictions.py
```

## MLflow Features Covered
This repository covers:
- **Experiment Tracking**: Logging hyperparameters, metrics, and artifacts.
- **Model Registry**: Storing, versioning, and managing ML models.
- **Model Deployment**: Serving and deploying trained models.
- **Artifact Storage**: Handling datasets, logs, and visualizations.
- **Hyperparameter Optimization**: Experimenting with tuning strategies.
- **Scaling MLflow**: Managing MLflow in large-scale projects.





