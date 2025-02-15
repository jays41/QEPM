import numpy as np
import pandas as pd

def standardize(df):
    # Convert raw factor data into Z-scores
    return (df - df.mean()) / df.std()

def compute_aggregate_z_score(z_scores, factor_group_weights):
    """
    Computes the aggregate Z-score for each stock.

    :param z_scores: DataFrame (rows: stocks, cols: factors) containing Z-scores
    :param factor_group_weights: Dict {factor_group: {factor: weight}} defining weights per factor group
    :return: DataFrame (rows: stocks, cols: factor_groups) of aggregate Z-scores
    """
    # Compute Factor Group Z-Scores
    factor_group_scores = {}
    for group, weights in factor_group_weights.items():
        relevant_factors = list(weights.keys())

        # Matrix multiplication: weighted sum of Z-scores
        factor_group_scores[group] = z_scores[relevant_factors] @ np.array(list(weights.values()))

    factor_group_df = pd.DataFrame(factor_group_scores, index=z_scores.index)

    # Compute Final Aggregate Z-Score
    factor_group_weights_vec = np.array([sum(weights.values()) for weights in factor_group_weights.values()])
    aggregate_z_scores = factor_group_df @ factor_group_weights_vec

    return aggregate_z_scores


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
z_scores = standardize(factor_data)
aggregate_z_scores = compute_aggregate_z_score(z_scores, factor_weights, group_weights)
print(aggregate_z_scores)