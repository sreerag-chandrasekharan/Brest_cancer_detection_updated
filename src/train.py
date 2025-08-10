import os, json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import yaml
import mlflow
import mlflow.sklearn

# --- Load params ---
with open("params.yaml") as f:
    params = yaml.safe_load(f)["train"]

# --- Load data ---
df = pd.read_csv('C:/Users/Sreerag/Documents/ML_chellange/Brest_Cancer_detection_updated/data/final/final.csv')   # adjust name
y = df['diagnosis']                        # 0/1 column
X = df.drop(columns=["diagnosis"])

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params["test_size"], random_state=params["random_state"], stratify=y)

# --- Model pipeline ---
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        C=params["C"],
        class_weight=params["class_weight"],
        max_iter=params["max_iter"],
        solver="lbfgs"
    ))
])


# --- MLflow setup ---
mlflow.set_experiment("cancer-detection") # adjust experiment name
with mlflow.start_run() as run:
    mlflow.log_params(params)

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "f1": float(f1_score(y_test, y_pred)),
        "accuracy": float(accuracy_score(y_test, y_pred))
    }
    mlflow.log_metrics(metrics)

# --- Save models ---
    import joblib
    os.makedirs("models", exist_ok=True)
    model_path = "Model/model.pkl"
    mlflow.sklearn.save_model(pipe, "Model/mlflow_model")  # MLflow format
    import joblib; joblib.dump(pipe, model_path)

#------- also write metrics for DVC
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

# ------Log artifacts to MLflow
    mlflow.log_artifact("metrics.json")
    mlflow.log_artifact(model_path)

    print("Run ID:", run.info.run_id)