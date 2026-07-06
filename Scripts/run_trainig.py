import pandas as pd
import mlflow

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import f1_score

from src.preprocess import split_data
from src.tune import tune_svm
from src.train import build_model, train_model, predict
from src.evaluate import evaluate_model
from src.save_model import save_model


def main():

    # -----------------------
    # MLflow setup
    # -----------------------
    mlflow.set_experiment("breast-cancer-svm")

    with mlflow.start_run():

        # -----------------------
        # Load data
        # -----------------------
        df = pd.read_csv('creditcard.csv')

        X = df[["V17","V14","V12","V10","V16","V3","V7","V11","V4","V18"]]
        y = df["Class"]

        # -----------------------
        # Split
        # -----------------------
        X_train, X_test, y_train, y_test = split_data(X, y)

        # -----------------------
        # Hyperparameter tuning
        # -----------------------
        tuning_results = tune_RF(X_train, y_train)

        best_params = tuning_results["best_params"]
        best_score = tuning_results["best_score"]

        mlflow.log_params(best_params)
        mlflow.log_metric("cv_f1", best_score)

        # -----------------------
        # Train final model
        # -----------------------
        model = build_model(**best_params)
        model = train_model(model, X_train, y_train)

        # -----------------------
        # Predict
        # -----------------------
        preds = predict(model, X_test)

        # -----------------------
        # Evaluation
        # -----------------------
        results = evaluate_model(model, X_test, y_test)

        mlflow.log_metric("accuracy", results["accuracy"])
        mlflow.log_metric("precision", results["precision"])
        mlflow.log_metric("recall", results["recall"])
        mlflow.log_metric("f1", results["f1"])

        # -----------------------
        # Save model
        # -----------------------
        save_model(model, "svm_optuna.pkl")

        mlflow.sklearn.log_model(model, "model")

        print("\nFinal Results:")
        print(results)


if __name__ == "__main__":
    main()
