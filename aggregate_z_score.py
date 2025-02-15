import numpy as np
import pandas as pd

def compute_aggregate_z_scores(factor_data, factor_groups, factor_weights, group_weights):
  # Built according to chapter 5.4.5
    """
    Computes the aggregate Z-score for each stock.

    :param factor_data: DataFrame where rows are stocks and columns are factor exposures.
    :param factor_groups: Dictionary mapping factor group names to lists of factors.
    :param factor_weights: Dictionary mapping factor group names to weight dictionaries for their factors.
    :param group_weights: Dictionary mapping factor group names to their weights.
    :return: Series with aggregate Z-scores for each stock.
    """

    z_scores = (factor_data - factor_data.mean()) / factor_data.std() # Compute Z-scores for each factor

    # Compute factor group Z-scores
    factor_group_z = {}
    for group, factors in factor_groups.items():
        if not all(f in z_scores.columns for f in factors):
            raise ValueError(f"Some factors in {group} are missing from data.")

        # Weighted sum of factor Z-scores
        weights = np.array([factor_weights[group][f] for f in factors])
        weights /= weights.sum()  # Ensure weights sum to 1
        factor_group_z[group] = z_scores[factors].dot(weights)

    factor_group_z_df = pd.DataFrame(factor_group_z)

    # Compute Aggregate Z-score for each stock
    group_weight_array = np.array([group_weights[g] for g in factor_group_z.keys()])
    group_weight_array /= group_weight_array.sum()  # Normalize group weights
    aggregate_z = factor_group_z_df.dot(group_weight_array)

    return aggregate_z


# Test data
factor_data = pd.DataFrame({
    "P/B": [1.5, 2.1, 1.2],
    "P/E": [15, 20, 10],
    "P/S": [2.5, 3.0, 1.8],
    "Net Profit Margin YoY": [5, 7, 6],
    "ROE YoY": [12, 15, 10]
}, index=["Stock A", "Stock B", "Stock C"])

factor_groups = {
    "Valuation": ["P/B", "P/E", "P/S"],
    "Profitability": ["Net Profit Margin YoY", "ROE YoY"]
}

factor_weights = {
    "Valuation": {"P/B": 0.4, "P/E": 0.4, "P/S": 0.2},
    "Profitability": {"Net Profit Margin YoY": 0.5, "ROE YoY": 0.5}
}

group_weights = {
    "Valuation": 0.6,
    "Profitability": 0.4
}

# Main
aggregate_z_scores = compute_aggregate_z_scores(factor_data, factor_groups, factor_weights, group_weights)
print(aggregate_z_scores)