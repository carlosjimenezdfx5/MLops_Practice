#=====================================#
#         Logging Functions           #
#         MLops PRactice DFX5         #
#=====================================#

import mlflow
from mlflow import MlflowClient

# Configurar la URI de rastreo
mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

# Nombre del modelo registrado
model_name = "ElasticNet"
# Alias que queremos asignar
alias = "champion"

# Buscar versiones del modelo registrado
model_versions = client.search_model_versions(f"name='{model_name}'")
if not model_versions:
    print(f"No se encontraron versiones para el modelo {model_name}.")
else:
    # Obtener la versión más reciente
    latest_version = max([int(mv.version) for mv in model_versions])

    # Asignar el alias 'champion' a la última versión del modelo usando tags
    client.set_model_version_tag(
        name=model_name,
        version=latest_version,
        key="alias",
        value=str(alias)  # Asegurarse de que el valor es una cadena
    )

    print(f"Alias '{alias}' asignado a la versión {latest_version} del modelo '{model_name}'.")



