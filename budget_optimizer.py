import pandas as pd


def reallocate_budget(campaign_perf, reduction_factor=0.3):
    """
    Automatically reallocates budget from low-performing
    campaigns to high-performing ones based on ROI.
    """

    campaign_perf = campaign_perf.copy()

    # Identify campaigns
    pause_df = campaign_perf[
        campaign_perf["recommendation"] == "Pause or Reduce Budget"
    ].copy()

    increase_df = campaign_perf[
        campaign_perf["recommendation"] == "Increase Budget"
    ].copy()

    # Initialize new budget
    campaign_perf["new_budget"] = campaign_perf["spent"]

    if pause_df.empty or increase_df.empty:
        return campaign_perf

    # Budget to reallocate
    pause_df["budget_cut"] = pause_df["spent"] * reduction_factor
    total_reallocated_budget = pause_df["budget_cut"].sum()

    # Distribute budget based on ROI weight
    increase_df["roi_weight"] = (
        increase_df["ROI"] / increase_df["ROI"].sum()
    )

    increase_df["budget_gain"] = (
        increase_df["roi_weight"] * total_reallocated_budget
    )

    # Apply budget cuts
    campaign_perf.loc[
        campaign_perf["campaign_id"].isin(pause_df["campaign_id"]),
        "new_budget"
    ] -= pause_df["budget_cut"].values

    # Apply budget gains
    campaign_perf.loc[
        campaign_perf["campaign_id"].isin(increase_df["campaign_id"]),
        "new_budget"
    ] += increase_df["budget_gain"].values

    return campaign_perf
