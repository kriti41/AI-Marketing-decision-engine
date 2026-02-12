import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load engineered data
df = pd.read_csv("data.csv")

# Recreate features (temporary – later we’ll modularize)
df["CTR"] = df["clicks"] / df["impressions"]
df = df.replace([float("inf"), -float("inf")], 0).fillna(0)

df["reporting_start"] = pd.to_datetime(df["reporting_start"], format="%d/%m/%Y")
df["day"] = df["reporting_start"].dt.day
df["month"] = df["reporting_start"].dt.month
df["day_of_week"] = df["reporting_start"].dt.dayofweek

df["age_group"] = df["age"].astype("category").cat.codes
df["gender"] = df["gender"].astype("category").cat.codes

X = df.drop(columns=[
    "ad_id", "reporting_start", "reporting_end", "CTR", "clicks"
]).copy()

for col in ["campaign_id", "fb_campaign_id", "age"]:
    X[col] = X[col].astype("category").cat.codes

y = df["CTR"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nTop Features Driving CTR:\n")
print(feature_importance.head(10))
import joblib

joblib.dump(model, "ctr_model.pkl")
print("Model saved as ctr_model.pkl")

feature_cols = list(X.columns)
joblib.dump(feature_cols, "feature_cols.pkl")
print("Feature columns saved as feature_cols.pkl")

