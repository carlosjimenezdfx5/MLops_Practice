import mlflow.pyfunc
import pandas as pd
import mlflow

# Configurar la URI de rastreo
mlflow.set_tracking_uri("http://127.0.0.1:5000")




model_uri = "models:/ElasticNet@champion"
model = mlflow.pyfunc.load_model(model_uri)


data = pd.read_csv("/Users/danieljimenez/Documents/Projects/Labor_Projects/globant/MLflow_Mentoring/chapter8/red-wine-data/test.csv")
predictions = model.predict(data)
print(predictions)
