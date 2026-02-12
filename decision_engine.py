import os
import pandas as pd
import joblib
from budget_optimizer import reallocate_budget  # your reallocation function
from llm import explain_decision

# -------------------------------
# Load model and features
# -------------------------------
model = joblib.load("ctr_model.pkl")
feature_cols = joblib.load("feature_cols.pkl")

# -------------------------------
# Load and prepare data
# -------------------------------
df = pd.read_csv("data.csv")

# Feature engineering
df["CTR"] = df["clicks"] / df["impressions"]
df = df.replace([float("inf"), -float("inf")], 0).fillna(0)

df["reporting_start"] = pd.to_datetime(df["reporting_start"], format="%d/%m/%Y", errors="coerce")
df["day"] = df["reporting_start"].dt.day
df["month"] = df["reporting_start"].dt.month
df["day_of_week"] = df["reporting_start"].dt.dayofweek

df["age_group"] = df["age"].astype("category").cat.codes
df["gender"] = df["gender"].astype("category").cat.codes

for col in ["campaign_id", "fb_campaign_id", "age"]:
    df[col] = df[col].astype("category").cat.codes

# Align features
X = df[feature_cols]

# -------------------------------
# Predict CTR
# -------------------------------
df["predicted_CTR"] = model.predict(X)

# -------------------------------
# Aggregate campaigns
# -------------------------------
campaign_perf = df.groupby("campaign_id").agg({
    "CTR": "mean",
    "predicted_CTR": "mean",
    "spent": "sum",
    "impressions": "sum"
}).reset_index()

# Compute ROI
campaign_perf["ROI"] = (campaign_perf["CTR"] * campaign_perf["impressions"]) / campaign_perf["spent"]

# -------------------------------
# Recommendations
# -------------------------------
def recommend_action(row):
    if row["ROI"] < campaign_perf["ROI"].quantile(0.25):
        return "Pause or Reduce Budget"
    elif row["ROI"] > campaign_perf["ROI"].quantile(0.75):
        return "Increase Budget"
    else:
        return "Monitor"

campaign_perf["recommendation"] = campaign_perf.apply(recommend_action, axis=1)

# -------------------------------
# Budget Reallocation
# -------------------------------
campaign_perf = reallocate_budget(campaign_perf)  # <-- NOW campaign_perf exists

# -------------------------------
# Output
# -------------------------------
print("\n💰 Budget Reallocation Plan:\n")
print(campaign_perf.sort_values("ROI", ascending=False)[
    ["campaign_id", "ROI", "spent", "new_budget", "recommendation"]
])

# AI explanations
campaign_perf["ai_explanation"] = campaign_perf.apply(explain_decision, axis=1)

for _, row in campaign_perf.head(3).iterrows():
    print("\nCampaign:", row["campaign_id"])
    print("Decision:", row["recommendation"])
    print("AI Explanation:", row["ai_explanation"])
    print("-" * 60)
