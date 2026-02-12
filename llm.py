def explain_decision(row):
    """
    Generate a human-readable explanation for campaign recommendations
    """

    if row["recommendation"] == "Increase Budget":
        return (
            f"This campaign is performing well with a CTR of {row['CTR']:.4f}. "
            f"The model predicts an even higher CTR ({row['predicted_CTR']:.4f}), "
            "indicating strong audience engagement. Increasing budget could maximize returns."
        )

    elif row["recommendation"] == "Pause or Reduce Budget":
        return (
            f"This campaign has a low CTR ({row['CTR']:.4f}) despite high spending. "
            "The model does not predict significant improvement, suggesting inefficient spend. "
            "Reducing or pausing the budget may prevent further losses."
        )

    else:
        return (
            f"This campaign shows stable performance with a CTR of {row['CTR']:.4f}. "
            "The predicted CTR is similar, indicating no immediate action is required. "
            "Continued monitoring is recommended."
        )
