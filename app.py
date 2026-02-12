import streamlit as st
import pandas as pd
import joblib
from llm import explain_decision  # or your AI explainer module

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="AI Marketing Decision Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title(" AI Marketing Decision Engine")

# -------------------------------
# Load model and features
# -------------------------------
MODEL_PATH = "ctr_model.pkl"
FEATURE_COLS_PATH = "feature_cols.pkl"
DATA_PATH = "data.csv"

model = joblib.load(MODEL_PATH)
feature_cols = joblib.load(FEATURE_COLS_PATH)

# -------------------------------
# Upload new data (optional)
# -------------------------------
st.sidebar.header("Upload Campaign Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv(DATA_PATH)

st.sidebar.markdown("---")

# -------------------------------
# Feature engineering (match training)
# -------------------------------
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

X = df[feature_cols]

# -------------------------------
# Prediction
# -------------------------------
df["predicted_CTR"] = model.predict(X)

# -------------------------------
# Campaign-level aggregation
# -------------------------------
campaign_perf = df.groupby("campaign_id").agg({
    "CTR": "mean",
    "predicted_CTR": "mean",
    "spent": "sum",
    "impressions": "sum"
}).reset_index()

campaign_perf["ROI"] = (campaign_perf["CTR"] * campaign_perf["impressions"]) / campaign_perf["spent"]

# -------------------------------
# Recommendation Logic
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
REDUCTION_FACTOR = 0.3
pause_df = campaign_perf[campaign_perf["recommendation"] == "Pause or Reduce Budget"].copy()
increase_df = campaign_perf[campaign_perf["recommendation"] == "Increase Budget"].copy()

pause_df["budget_cut"] = pause_df["spent"] * REDUCTION_FACTOR
total_reallocated_budget = pause_df["budget_cut"].sum()

if not increase_df.empty:
    increase_df["roi_weight"] = increase_df["ROI"] / increase_df["ROI"].sum()
    increase_df["budget_gain"] = increase_df["roi_weight"] * total_reallocated_budget
else:
    increase_df["budget_gain"] = 0

campaign_perf["new_budget"] = campaign_perf["spent"]
campaign_perf.loc[campaign_perf["campaign_id"].isin(pause_df["campaign_id"]), "new_budget"] -= pause_df["budget_cut"].values
campaign_perf.loc[campaign_perf["campaign_id"].isin(increase_df["campaign_id"]), "new_budget"] += increase_df["budget_gain"].values

# -------------------------------
# AI Explanations
# -------------------------------
st.subheader("Budget Reallocation Plan")
st.dataframe(
    campaign_perf.sort_values("ROI", ascending=False)[
        ["campaign_id", "ROI", "spent", "new_budget", "recommendation"]
    ]
)

st.subheader(" Campaign AI Explanations (Top 5)")
campaign_perf["ai_explanation"] = campaign_perf.apply(explain_decision, axis=1)
for _, row in campaign_perf.head(5).iterrows():
    st.markdown(f"**Campaign {row['campaign_id']} — {row['recommendation']}**")
    st.write(row["ai_explanation"])
    st.markdown("---")
