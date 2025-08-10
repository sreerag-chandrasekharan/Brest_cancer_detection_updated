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
Logreg_pipline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        C=params["C"],
        class_weight=params["class_weight"],
        max_iter=params["max_iter"],
        solver="lbfgs"
    ))
])

xgb_pipeline = Pipeline([
    ("scaler", StandardScaler()),  # optional, can be omitted if you want
    ("clf", XGBClassifier(
        n_estimators=params.get("n_estimators", 100),
        learning_rate=params.get("learning_rate", 0.1),
        max_depth=params.get("max_depth", 6),
        random_state=params.get("random_state", 42),
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=params.get("scale_pos_weight", 1),
        # add any other params you want here
    ))
])

# --- MLflow setup ---
mlflow.set_experiment("cancer-detection") # adjust experiment name
with mlflow.start_run() as run:
    mlflow.log_params(params)

    Logreg_pipline.fit(X_train, y_train)
    y_pred = Logreg_pipline.predict(X_test)
    y_proba = Logreg_pipline.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "f1": float(f1_score(y_test, y_pred)),
        "accuracy": float(accuracy_score(y_test, y_pred))
    }
    mlflow.log_metrics(metrics)

    xgb_pipeline.fit(X_train, y_train)
    y_pred_xgb = xgb_pipeline.predict(X_test)
    y_proba_xgb = xgb_pipeline.predict_proba(X_test)[:, 1]
    metrics_xgb = {
        "roc_auc": float(roc_auc_score(y_test, y_proba_xgb)),
        "f1": float(f1_score(y_test, y_pred_xgb)),
        "accuracy": float(accuracy_score(y_test, y_pred_xgb))
    }
    mlflow.log_metrics(metrics_xgb)

# --- Save and log models ---
import joblib
def save_and_log_models(loreg_pipe, xgb_pipe):
    os.makedirs("models", exist_ok=True)

    # Logistic Regression model
    loreg_pkl_path = "models/loreg_model.pkl"
    loreg_mlflow_path = "models/mlflow_loreg_model"
    joblib.dump(loreg_pipe, loreg_pkl_path)
    mlflow.sklearn.save_model(loreg_pipe, loreg_mlflow_path)
    mlflow.sklearn.log_model(loreg_pipe, artifact_path="loreg_model")
    print(f"Logistic Regression saved as pickle: {loreg_pkl_path}")
    print(f"Logistic Regression saved in MLflow format: {loreg_mlflow_path}")

    # XGBoost model
    xgb_pkl_path = "models/xgb_model.pkl"
    xgb_mlflow_path = "models/mlflow_xgb_model"
    joblib.dump(xgb_pipe, xgb_pkl_path)
    mlflow.sklearn.save_model(xgb_pipe, xgb_mlflow_path)
    mlflow.sklearn.log_model(xgb_pipe, artifact_path="xgb_model")
    print(f"XGBoost saved as pickle: {xgb_pkl_path}")
    print(f"XGBoost saved in MLflow format: {xgb_mlflow_path}")

#------- also write metrics for DVC
    with open("metrics_log.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open("metrics_xg.json", "w") as f:
        json.dump(metrics, f, indent=2)

# ------Log artifacts to MLflow
    mlflow.log_artifact("metrics_log.json")
    mlflow.log_artifact(loreg_pkl_path)

    mlflow.log_artifact("metrics_xg.json")
    mlflow.log_artifact(xgb_pkl_path)

    print("Run ID:", run.info.run_id)