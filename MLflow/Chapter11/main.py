#=================================================#
#                 MLflow Integration              #
#                 LLMOps on AWS  DFX5             #
#=================================================#

"""
This script demonstrates the integration of MLflow with AWS for managing
Large Language Models (LLMs). It includes model fine-tuning, tracking,
and deployment using AWS services.
"""

import os
import mlflow
import boto3
import torch
import mlflow.sagemaker as mfs
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

# AWS Configuration -------
s3_bucket = 'bucket-project'
region_name = 'us-east-1'  # Replace with  AWS region
os.environ['MLFLOW_S3_ENDPOINT_URL'] = f'https://s3.{region_name}.amazonaws.com'
mlflow.set_tracking_uri(f'http://your-mlflow-tracking-server')

# Model and Tokenizer Initialization ------
model_name = 'gpt4o'  
model = AutoModelForCausalLM.from_pretrained(model_name)

# Prepare Dataset ----------
def load_dataset(file_path, tokenizer, block_size=128):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )
    return dataset

train_dataset = load_dataset('path_to_your_training_data.txt', tokenizer)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Training Arguments --------
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
)

# Initialize Trainer -------
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# Fine-tuning the Model with MLflow Tracking -------
with mlflow.start_run() as run:
    trainer.train()
    
    # Save and log the model -------
    model_path = "model"
    trainer.save_model(model_path)
    mlflow.log_artifacts(model_path, artifact_path="model")
    
    # Log model to MLflow Model Registry
    mlflow.pytorch.log_model(model, "model", registered_model_name="GPT-2-Finetuned")

# Deploying the Model on AWS SageMaker ------

app_name = "gpt4o-finetuned"
model_uri = f"runs:/{run.info.run_id}/model"
image_uri = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.9.0-cpu-py38-ubuntu20.04"  # Replace with the appropriate image URI
role_arn = "arn:aws:iam::your-account-id:role/sagemaker-role"

mfs.deploy(
    app_name=app_name,
    model_uri=model_uri,
    region_name=region_name,
    mode="create",
    execution_role_arn=role_arn,
    image_url=image_uri,
    instance_type="ml.m5.large",
    s3_bucket=s3_bucket,
    timeout_seconds=3600,
    synchronous=True
)
