import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.data_loader import load_data
from src.preprocessing import clean_data
from src.feature_engineering import create_features
from src.model_training import build_model
from src.evaluator import evaluate_model
from src.kpi_calculator import calculate_kpis
from src.visualizer import plot_confusion_matrix, plot_roc_curve
from src.feature_config import get_model_features


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "DataCoSupplyChainDataset.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "delivery_risk_model.pkl")
COMPARISON_PATH = os.path.join(BASE_DIR, "models", "model_comparison.csv")
FIGURE_DIR = os.path.join(BASE_DIR, "reports", "figures")


def main():
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "data", "processed"), exist_ok=True)

    print("PROJECT BASE DIR:")
    print(BASE_DIR)

    print("MODEL WILL SAVE TO:")
    print(MODEL_PATH)

    df = load_data(RAW_DATA_PATH)

    if df is None:
        print("❌ Data not loaded")
        return

    df = clean_data(df)
    df = create_features(df)

    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print("✅ Cleaned data saved")
    print("CLEANED DATA SAVED TO:")
    print(PROCESSED_DATA_PATH)

    kpis = calculate_kpis(df)

    print("\n📊 KPIs:")
    for key, value in kpis.items():
        print(f"{key}: {value}")

    y = df["Late_delivery_risk"]
    X = get_model_features(df)

    print("\nTRAINING FEATURE SHAPE:")
    print(X.shape)

    print("\nTRAINING FEATURE COLUMNS COUNT:")
    print(len(X.columns))

    print("\nTRAINING FEATURES:")
    print(X.columns.tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = build_model(X_train)

    model.fit(X_train, y_train)

    print("\nMODEL RAW INPUT FEATURES:")
    print(model.feature_names_in_)
    print("MODEL RAW INPUT FEATURE COUNT:")
    print(len(model.feature_names_in_))

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results = evaluate_model(y_test, y_pred)

    print("\n🤖 Random Forest Results:")
    for key, value in results.items():
        print(f"{key}: {round(value, 4)}")

    print("\n📊 Training Logistic Regression Benchmark...")

    log_preprocessor = clone(model.named_steps["preprocessor"])

    log_model = Pipeline(
        steps=[
            ("preprocessor", log_preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=300,
                    solver="liblinear",
                    random_state=42
                )
            )
        ]
    )

    X_train_log = X_train.sample(
        n=min(10000, len(X_train)),
        random_state=42
    )

    y_train_log = y_train.loc[X_train_log.index]

    log_model.fit(X_train_log, y_train_log)

    log_pred = log_model.predict(X_test)

    log_results = {
        "Accuracy": accuracy_score(y_test, log_pred),
        "Precision": precision_score(y_test, log_pred),
        "Recall": recall_score(y_test, log_pred),
        "F1 Score": f1_score(y_test, log_pred)
    }

    comparison_df = pd.DataFrame(
        [
            {
                "Model": "Logistic Regression",
                "Accuracy": round(log_results["Accuracy"], 4),
                "Precision": round(log_results["Precision"], 4),
                "Recall": round(log_results["Recall"], 4),
                "F1 Score": round(log_results["F1 Score"], 4)
            },
            {
                "Model": "Random Forest",
                "Accuracy": round(results["Accuracy"], 4),
                "Precision": round(results["Precision"], 4),
                "Recall": round(results["Recall"], 4),
                "F1 Score": round(results["F1 Score"], 4)
            }
        ]
    )

    comparison_df.to_csv(COMPARISON_PATH, index=False)

    print("\n📊 Model Comparison:")
    print(comparison_df)

    joblib.dump(model, MODEL_PATH)

    print("\n✅ Model saved successfully!")
    print("MODEL SAVED TO:")
    print(os.path.abspath(MODEL_PATH))

    loaded_model = joblib.load(MODEL_PATH)
    print("LOADED SAVED MODEL FEATURE COUNT:")
    print(len(loaded_model.feature_names_in_))

    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_prob)

    print("✅ Confusion Matrix saved")
    print("✅ ROC Curve saved")

    try:
        rf = model.named_steps["classifier"]
        importances = rf.feature_importances_

        feature_names = [f"Feature_{i}" for i in range(len(importances))]

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        })

        importance_df = (
            importance_df
            .sort_values("Importance", ascending=False)
            .head(15)
        )

        plt.figure(figsize=(10, 6))
        plt.barh(importance_df["Feature"], importance_df["Importance"])
        plt.gca().invert_yaxis()
        plt.title("Top 15 Feature Importances")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, "feature_importance.png"))
        plt.close()

        print("✅ Feature Importance saved")

    except Exception as e:
        print("⚠️ Feature Importance skipped:", e)

    print("✅ Model comparison saved")


if __name__ == "__main__":
    main()