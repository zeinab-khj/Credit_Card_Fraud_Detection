from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


def build_model(max_depth=None, n_estimators=100, Random_state=42):
    model = Pipeline([
        ("RandomForest", RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, Random_state=Random_state)
    ])
    return model


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    return model.predict(X_test)
