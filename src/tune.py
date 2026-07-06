import optuna
import mlflow
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassfier

def tuen_RF(X_train, y_train, n_trials=30):
  def objectiv(trail):
    param = {
        "max_depth": trail.suggest_int("max_depth", None, 10),
        "n_estimators": trail.suggest_int("n_estimators", 50, 200),
        "random_state": 42
    }
  
   
    pipeline = Pipeline([
        ("RandomForest", RandomForestClassifier(**param))
    ])
    
    score = cross_val_score(
        estimator=pipeline,
        X=X_train,
        y=y_train,
        cv=5,
        scoring="f1",
        n_jobs=-1
    ).mean()

    with mlflow.start_run(nested=True):
    mlflow.log_param("C", C)
    mlflow.log_param("kernel", kernel)
    mlflow.log_param("gamma", gamma)
    mlflow.log_metric("f1", score)


    return score

study = optuna.create_study(direction="maximize")

study.optimize(
    objective,
    n_trials=n_trials,
    show_progress_bar=True
)

return {
  "best_params": study.best_params,
  "best_score": study.best_value,
  "best_trial": study.best_trial,
  "study": study,
}
