import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import os

df = pd.read_csv("data/crop_data.csv")
X = df[['N','P','K','temperature','humidity','ph','rainfall']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlruns_dir = os.path.join(os.getcwd(), "mlruns")
mlflow.set_tracking_uri(f"file://{mlruns_dir}")
mlflow.set_experiment("Crop_Prediction_Training")

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")

    mlflow.log_metric("accuracy", acc)
    mlflow.log_params({"n_estimators":100,"random_state":42})

    os.makedirs("mlflow_models", exist_ok=True)
    model_path = "mlflow_models/crop_model.pkl"
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)