import numpy as np
import pandas as pd
import statsmodels.api as sm
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple
from scipy import stats

def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized standardization using numpy"""
    return (df - df.mean()) / df.std()

def compute_aggregate_z_score(
    z_scores: pd.DataFrame,
    factor_group_weights: Dict[str, Dict[str, float]],
    group_weights: Dict[str, float]
) -> pd.DataFrame:
    """
    Optimized computation of aggregate Z-scores using vectorized operations
    """
    # Pre-compute weight matrices
    factor_weights_matrix = {}
    for group, weights in factor_group_weights.items():
        weight_array = np.zeros(len(z_scores.columns))
        for i, col in enumerate(z_scores.columns):
            weight_array[i] = weights.get(col, 0)
        factor_weights_matrix[group] = weight_array

    # Vectorised computation of factor group scores
    group_scores = {
        group: z_scores.values @ weights
        for group, weights in factor_weights_matrix.items()
    }
    
    factor_group_df = pd.DataFrame(group_scores, index=z_scores.index)
    group_weight_array = np.array([group_weights[group] for group in factor_group_df.columns])
    
    # Vectorised final computation
    aggregate_z_scores = factor_group_df.values @ group_weight_array
    return pd.DataFrame(aggregate_z_scores, index=z_scores.index, columns=['z_score'])


def process_stock_chunk(
    chunk_data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]
) -> Dict[str, Dict[str, float]]:
    """Process a chunk of stocks for parallel computation"""
    stock_returns, factor_returns, betas, aggregate_z_scores, stock_chunk = chunk_data
    
    # Pre-compute expected returns for the chunk
    expected_returns = factor_returns @ betas[stock_chunk]
    alpha = stock_returns[stock_chunk] - expected_returns
    
    chunk_results = {}
    for stock in stock_chunk:
        y = alpha[stock].iloc[1:].values  # Convert to numpy array
        X = aggregate_z_scores['z_score'].iloc[1:].values  # Convert to numpy array
        
        if len(y) < 2:
            chunk_results[stock] = {"gamma": None, "delta": None, "p-value": None}
            continue

        # Use scipy for faster computation of basic stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
        
        chunk_results[stock] = {
            "gamma": intercept,
            "delta": slope,
            "p-value": p_value
        }
    
    return chunk_results

def estimate_alpha(
    stock_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    betas: pd.DataFrame,
    aggregate_z_scores: pd.DataFrame,
    n_jobs: int = None
) -> Dict[str, Dict[str, float]]:
    """
    Parallelized alpha estimation using optimized computations
    """
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)
    
    # Use a single process for small datasets
    if len(stock_returns.columns) < 4:
        n_jobs = 1
 
    # Split stocks into chunks for parallel processing
    stocks = list(stock_returns.columns)
    chunk_size = max(1, len(stocks) // n_jobs)
    stock_chunks = [stocks[i:i + chunk_size] for i in range(0, len(stocks), chunk_size)]
    
    # Prepare data for parallel processing
    chunk_data = [(stock_returns, factor_returns, betas, aggregate_z_scores, chunk) 
                  for chunk in stock_chunks]
    
    if n_jobs > 1:
        # Process chunks in parallel
        with Pool(n_jobs) as pool:
            chunk_results = pool.map(process_stock_chunk, chunk_data)
    else:
        # Process in single thread
        chunk_results = [process_stock_chunk(data) for data in chunk_data]
    
    # Combine results
    final_results = {}
    for chunk_result in chunk_results:
        final_results.update(chunk_result)
    
    return final_results

def run_analysis(factor_data, factor_groups, factor_weights, group_weights, 
                stock_returns, factor_returns, betas, n_jobs=None):
    """Main analysis function with performance monitoring"""
    import time
    
    start_time = time.time()
    
    # Standardization
    z_scores = standardize(factor_data)
    
    # Compute aggregate Z-scores
    aggregate_z_scores = compute_aggregate_z_score(z_scores, factor_weights, group_weights)
    
    # Estimate alpha
    alpha_results = estimate_alpha(stock_returns, factor_returns, betas, 
                                 aggregate_z_scores, n_jobs=n_jobs)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return aggregate_z_scores, alpha_results, execution_time



# Test data
dates = pd.date_range('2024-01-01', periods=3)

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

stock_returns = pd.DataFrame({
    "Stock A": [0.02, 0.03, 0.01],
    "Stock B": [0.05, 0.02, -0.01],
    "Stock C": [-0.01, 0.01, 0.03]
}, index=dates)

factor_returns = pd.DataFrame({
    "Factor 1": [0.01, 0.02, 0.01],
    "Factor 2": [0.02, -0.01, 0.00]
}, index=dates)

betas = pd.DataFrame({
    "Stock A": [1.2, 0.8],
    "Stock B": [0.9, 1.1],
    "Stock C": [1.5, 0.7]
}, index=["Factor 1", "Factor 2"])

# Analysis
aggregate_z_scores, alpha_results, execution_time = run_analysis(
    factor_data=factor_data,
    factor_groups=factor_groups,
    factor_weights=factor_weights,
    group_weights=group_weights,
    stock_returns=stock_returns,
    factor_returns=factor_returns,
    betas=betas,
    n_jobs=1  # Using single process for small dataset
)


# Results
print(f"\nExecution time: {execution_time:.4f} seconds")
print("\nStocks Ranked by Aggregate Z-score:")
print(aggregate_z_scores.sort_values('z_score', ascending=False))
print("\nAlpha estimation results:")
for stock, results in alpha_results.items():
    print(f"\n{stock}:")
    print(f"  Gamma (intercept): {results['gamma']:.6f}")
    print(f"  Delta (slope): {results['delta']:.6f}")
    print(f"  P-value: {results['p-value']:.6f}")