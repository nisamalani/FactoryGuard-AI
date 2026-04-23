import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def train_baseline(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model, None


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    except:
        print("XGBoost not installed, using RandomForest instead")
        return train_random_forest(X_train, y_train)

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, scaler=None):
    if scaler:
        X_test = scaler.transform(X_test)

    preds = model.predict(X_test)
    print("\n📊 Evaluation Report:")
    print(classification_report(y_test, preds))


def save_model(model, path):
    joblib.dump(model, path)