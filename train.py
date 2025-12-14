import pandas as pd
import joblib
from datetime import datetime
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import gcsfs
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
from pprint import pprint
from evidently import DataDefinition
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
import os
import pickle
from google.cloud import storage

def upload_to_gcs(bucket_name, source_path, dest_blob):
    """Uploads a file to the specified GCS bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(dest_blob)
    blob.upload_from_filename(source_path)
    print(f"✅ Uploaded model to gs://{bucket_name}/{dest_blob}")

     

def train_model_with_feast():
    data = pd.read_csv('data/iris.csv')
    train, test = train_test_split(data, test_size = 0.4, stratify = data['species'], random_state = 42)
    X_train = train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_train = train.species
    X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_test = test.species
    mlflow.set_tracking_uri("http://10.128.0.2:8100")
    client = MlflowClient(mlflow.get_tracking_uri())
    all_experiments = client.search_experiments()
    print(all_experiments)
    mlflow.set_experiment("IRIS Classifier Test: MLFlow")
    params = {
    "max_depth": 2,
    "random_state": 1
    }
    mod_dt = DecisionTreeClassifier(**params)
    mod_dt.fit(X_train, y_train)
    prediction = mod_dt.predict(X_test)
    accuracy_score = metrics.accuracy_score(prediction, y_test)
    print(accuracy_score)
    filename = 'artifacts/model.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(mod_dt, file)
    joblib.dump(mod_dt, "artifacts/model.joblib")
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy_score)
        mlflow.set_tag("Training info", "Decision Tree First Run")
        signature = infer_signature(X_train, mod_dt.predict(X_train))

        model_info = mlflow.sklearn.log_model(
            sk_model = mod_dt,
            artifact_path = "iris_model",
            signature = signature,
            input_example = X_train,
            registered_model_name = "IRIS-classifier-dt"
        )
    upload_to_gcs("artifacts2-graphite-dynamo-473907-c1", "artifacts/model.pkl", "production_models/model.pkl")
if __name__ == "__main__":
    train_model_with_feast()