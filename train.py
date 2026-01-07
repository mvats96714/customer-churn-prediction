import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# 1. Load data
df = pd.read_csv("data/churn.csv")

# 2. Drop ID column
df.drop("customerID", axis=1, inplace=True)

# 3. Target encoding
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# 4. Fix missing values
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# 5. One-hot encode categorical columns
df = pd.get_dummies(df, drop_first=True)

# 6. Split X and y
X = df.drop("Churn", axis=1)
y = df["Churn"]

feature_names = X.columns
joblib.dump(feature_names, "features.pkl")


# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 9. Train models
from xgboost import XGBClassifier

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])

# XGBoost
xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    eval_metric="logloss"
)
xgb.fit(X_train, y_train)
xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])

print("Logistic Regression ROC-AUC:", lr_auc)
print("XGBoost ROC-AUC:", xgb_auc)

# 10. Select best model
best_model = xgb if xgb_auc > lr_auc else lr

# 11. Save best model and scaler
joblib.dump(best_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Training complete. Best model saved.")


