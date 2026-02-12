import pandas as pd

# Load data
df = pd.read_csv("data.csv")

# Create target
df["CTR"] = df["clicks"] / df["impressions"]
df = df.replace([float("inf"), -float("inf")], 0).fillna(0)

# Date features
df["reporting_start"] = pd.to_datetime(df["reporting_start"], format="%d/%m/%Y")
df["day"] = df["reporting_start"].dt.day
df["month"] = df["reporting_start"].dt.month
df["day_of_week"] = df["reporting_start"].dt.dayofweek

# Encode categorical features
df["age_group"] = df["age"].astype("category").cat.codes
df["gender"] = df["gender"].astype("category").cat.codes

# Prepare ML dataset (avoid leakage)
df_model = df.drop(columns=[
    "ad_id",
    "reporting_start",
    "reporting_end",
    "CTR",
    "clicks"
])

X = df_model.copy()
y = df["CTR"]

# Encode remaining categorical columns
for col in ["campaign_id", "fb_campaign_id", "age"]:
    X[col] = X[col].astype("category").cat.codes

# Sanity check
print("X shape:", X.shape)
print("\nData types:\n", X.dtypes)
print("\nTarget stats:\n", y.describe())
