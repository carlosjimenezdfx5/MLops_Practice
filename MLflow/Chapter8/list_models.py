#=====================================#
#         Logging Functions           #
#         MLops PRactice DFX5         #
#=====================================#
from mlflow import MlflowClient
import mlflow

# Configurar la URI de rastreo
mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

# Listar todos los modelos registrados
registered_models = client.search_registered_models()
if not registered_models:
    print("No hay modelos registrados.")
else:
    for model in registered_models:
        print(f"Model Name: {model.name}")
        # Listar versiones de cada modelo
        model_versions = client.search_model_versions(f"name='{model.name}'")
        for mv in model_versions:
            print(f"  Version: {mv.version}, Stage: {mv.current_stage}, Run ID: {mv.run_id}")

