import sys
sys.path.insert(0, "src")
from data_loader import load_data, clean_data
from feature_engineering import add_rolling_features, get_feature_columns
from model import train_baseline, train_xgboost, evaluate_model, explain_with_shap, save_model
from sklearn.model_selection import train_test_split

print("Step 1: Loading data...")
df = load_data("data/sensor_data.csv")
df = clean_data(df)
print(f"Data loaded: {df.shape}")

print("Step 2: Feature engineering...")
df = add_rolling_features(df)
feature_cols = get_feature_columns(df)
X = df[feature_cols].values
y = df["failure"].values
print(f"Features created: {len(feature_cols)}")

print("Step 3: Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Step 4: Training baseline...")
lr_model, scaler = train_baseline(X_train, y_train)
evaluate_model(lr_model, X_test, y_test, scaler=scaler)

print("Step 5: Training XGBoost...")
xgb_model = train_xgboost(X_train, y_train)
evaluate_model(xgb_model, X_test, y_test)

print("Step 6: SHAP explanations...")
explain_with_shap(xgb_model, X_test[:100], feature_cols)

print("Step 7: Saving model...")
save_model(xgb_model, "model.pkl")
print("TRAINING COMPLETE!")
