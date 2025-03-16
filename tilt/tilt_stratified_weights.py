import pandas as pd
import numpy as np
import cvxpy as cp

# Define tilting factor vector (randomly generated for demonstration)
np.random.seed(42)
gammas = np.random.uniform(-1, 1, 100)  # 100 stocks with tilting factors between -1 and 1

def tilt_calculate(taus):

    
    
    return

def get_stratified_weights(stock_data, expected_returns, cov_matrix, gammas, sectors_array, n_betas, target_annual_risk):

    # Restrict tau to positive effect
    betas = gammas[:n_betas]  # First X values are betas
    taus = gammas[n_betas:]  # Remaining values are taus

    n = len(expected_returns)
    
    # Convert annual risk target to daily
    target_risk = (1 + target_annual_risk) ** (1 / 252) - 1

    # Small tolerance for equality constraints
    epsilon = 1e-6

    # Identify unique sectors
    unique_sectors = np.unique(sectors_array)
    num_sectors = len(unique_sectors)

    # Calculate sector proportions and indices
    sector_proportions = {}
    sector_indices = {}
    for sector_id in unique_sectors:
        indices = np.where(sectors_array == sector_id)[0]
        sector_indices[sector_id] = indices
        sector_proportions[sector_id] = len(indices) / n  # Proportion of stocks in this sector

    # Define optimization variable (weights)
    w = cp.Variable(n)

    # Compute Cholesky decomposition of covariance matrix for risk constraint
    cov_chol = np.linalg.cholesky(cov_matrix)

    # Objective: Maximize expected return
    objective = cp.Maximize(cp.sum(cp.multiply(expected_returns, w)))

    # Constraints
    constraints = [
        cp.norm(cp.sum(w), 2) <= epsilon,  # Dollar neutrality
        cp.norm(cp.sum(cp.multiply(betas, w)), 2) <= epsilon,  # Beta neutrality
        cp.norm(cov_chol @ w, 2) <= target_risk,  # Risk constraint (SOC)
        cp.norm(w, 1) <= 2,  # Gross exposure limit
        w >= -0.2,  # Short position limit
        w <= 0.2  # Long position limit
    ]

    # Sector neutrality and proportional exposure constraints
    for sector_id in unique_sectors:
        indices = sector_indices[sector_id]
        sector_weight = w[indices]

        # Ensure net exposure within each sector is zero
        constraints.append(cp.sum(sector_weight) == 0)

        # Ensure gross exposure is proportional to sector size
        max_sector_exposure = sector_proportions[sector_id] * 2  
        constraints.append(cp.sum(cp.abs(sector_weight)) <= max_sector_exposure)

    # Compute tilt values using external function
    # Compute tilt values using external function
    tilt_aggregate = tilt_calculate(taus)

    # Implement tilting constraint: ensure long and short portfolios balance tilting exposure
    long_weights = cp.pos(w)  # Long positions only
    short_weights = cp.neg(w)  # Short positions only

    # sum of long weights * tau ≈ tilt * (sum of short weights * tau) with tolerance
    constraints.append(cp.norm(cp.sum(cp.multiply(long_weights, taus)) - tilt_aggregate * cp.sum(cp.multiply(short_weights, taus)), 2) <= epsilon)

"""

# Try solving the problem with improved solver parameters
    try:
        # Try SCS with better parameters
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS, eps=1e-8, max_iters=10000000000, alpha=1.8)
        print("Solved with SCS using improved parameters")
    except Exception as e:
        print(f"SCS solver failed: {e}")
        try:
            # Try OSQP if SCS fails
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8, max_iter=10000)
            print("Solved with OSQP")
        except Exception as e:
            print(f"OSQP solver failed: {e}")
            try:
                # Try ECOS if OSQP fails
                problem = cp.Problem(objective, constraints)
                problem.solve(solver=cp.ECOS, abstol=1e-8, reltol=1e-8, max_iters=1000)
                print("Solved with ECOS")
            except Exception as e:
                print(f"ECOS solver failed: {e}")
                print("All standard solvers failed. Trying alternative approach.")
                
                # Final attempt with CVXOPT with better parameters
                problem = cp.Problem(objective, constraints)
                problem.solve(solver=cp.CVXOPT, abstol=1e-8, reltol=1e-8)
                print("Solved with CVXOPT using improved parameters")

    # Output optimized weights
    if problem.status == "optimal" or problem.status == "optimal_inaccurate":
        optimized_weights = w.value
        print("\nOptimization successful!")
        print("Status:", problem.status)
        # print("Optimized weights:", optimized_weights)
        
        # Calculate the long and short positions
        long_positions = np.sum(optimized_weights[optimized_weights > 0])
        short_positions = np.sum(optimized_weights[optimized_weights < 0])
        
        # Calculate actual risk using original covariance matrix
        portfolio_variance = optimized_weights @ cov_matrix @ optimized_weights
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Calculating annual expected returns
        expected_daily_returns = np.sum(expected_returns * optimized_weights)
        annual_return = (1 + expected_daily_returns) ** 252 - 1
        annual_volatility = portfolio_volatility * np.sqrt(252)
        annual_sharpe = annual_return / annual_volatility
        
        # Verification
        # print("\nVerification:")
        # print(f"Sum of weights (dollar neutrality, should be 0): {np.sum(optimized_weights):.8f}")
        # print(f"Long positions: {long_positions:.6f}")
        # print(f"Short positions: {short_positions:.6f}")
        # print(f"Beta neutrality (should be 0): {np.sum(betas * optimized_weights):.8f}")
        # print(f"Expected return: {expected_daily_returns:.6f} ({100 * expected_daily_returns:.6f} %) per day")
        # print(f"Portfolio variance: {portfolio_variance:.6f}")
        # print(f"Portfolio volatility: {portfolio_volatility:.6f}")
        # print(f"Target volatility: {target_risk:.6f}")
        # print(f"Gross exposure: {np.sum(np.abs(optimized_weights)):.6f}")
        
        # print("\nAnnualised Metrics:")
        # print(f"Annual expected return: {annual_return:.6f} ({100 * annual_return:.2f}%)")
        # print(f"Annual volatility: {annual_volatility:.6f} ({100 * annual_volatility:.2f}%)")
        # print(f"Annual Sharpe ratio: {annual_sharpe:.4f}")
        
        # Sector-wise analysis
        # print("\nSector-wise Analysis:")
        for sector_id in unique_sectors:
            indices = sector_indices[sector_id]
            sector_weights = optimized_weights[indices]
            sector_long_positions = np.sum(sector_weights[sector_weights > 0])
            sector_short_positions = np.sum(sector_weights[sector_weights < 0])
            sector_exposure = np.sum(np.abs(sector_weights))
            max_allowed = sector_proportions[sector_id] * 2
            
            # print(f"\nSector {sector_id}:")
            # print(f"  Number of stocks: {len(indices)}")
            # print(f"  Sum of weights (should be 0): {np.sum(sector_weights):.8f}")
            # print(f"  Long positions: {sector_long_positions:.6f}")
            # print(f"  Short positions: {sector_short_positions:.6f}")
            # print(f"  Gross exposure: {sector_exposure:.6f} (max allowed: {max_allowed:.6f})")
            # print(f"  Exposure constraint satisfied: {sector_exposure <= max_allowed + 1e-6}")

    else:
        print("Optimization failed with status:", problem.status)

    # Additional analysis of the portfolio
    if problem.status == "optimal" or problem.status == "optimal_inaccurate":
        # Count positions
        num_long = np.sum(optimized_weights > 0.001)  # Positions > 0.1%
        num_short = np.sum(optimized_weights < -0.001)  # Positions < -0.1%
        num_neutral = np.sum(np.abs(optimized_weights) <= 0.001)  # Near-zero positions
        
        # print("\nPortfolio Structure:")
        # print(f"Number of long positions (>0.1%): {num_long}")
        # print(f"Number of short positions (<-0.1%): {num_short}")
        # print(f"Number of near-zero positions: {num_neutral}")
        
        # Calculate dollar-neutrality deviation
        print(f"Dollar-neutrality deviation: {np.abs(long_positions + short_positions):.8f}")
        
        # Calculate correlation with market factors
        if np.std(betas @ optimized_weights) > 0:
            market_correlation = np.corrcoef(betas, optimized_weights)[0,1]
            # print(f"Correlation with market factor: {market_correlation:.8f}")
        
        # Print constraint satisfaction levels
        # print("\nConstraint Verification:")
        # print(f"Dollar neutrality error: {np.abs(np.sum(optimized_weights)):.8f}")
        # print(f"Beta neutrality error: {np.abs(np.sum(betas * optimized_weights)):.8f}")
        # print(f"Risk constraint: {portfolio_volatility:.6f} <= {target_risk:.6f} is {portfolio_volatility <= target_risk + 1e-6}")
        # print(f"Gross exposure: {np.sum(np.abs(optimized_weights)):.6f} <= 2 is {np.sum(np.abs(optimized_weights)) <= 2 + 1e-6}")
        # print(f"Position limits satisfied: {np.all(optimized_weights >= -0.1-1e-6) and np.all(optimized_weights <= 0.1+1e-6)}")
        



    # Create a DataFrame with stock tickers, sectors, and weights
    if problem.status == "optimal" or problem.status == "optimal_inaccurate":
        # First, check the lengths to debug
        # print(f"Number of unique tickers: {len(stock_data['ticker'].unique())}")
        # print(f"Length of sectors_array: {len(sectors_array)}")
        # print(f"Length of optimized_weights: {len(optimized_weights)}")
        
        # Get the correct tickers in the same order as sectors_array and optimized_weights
        # We need to ensure they all have the same length and order
        tickers = stock_data['ticker'].unique()
        
        # Make sure we use the same number of items for all arrays
        n_stocks = min(len(tickers), len(sectors_array), len(optimized_weights))
        
        # Create a mapping from sectors to names if possible
        sector_name_mapping = {}
        sector_data = stock_data.drop_duplicates('ticker')[['ticker', 'sector']]
        sector_dict = {i: sector for i, sector in enumerate(sector_data['sector'].unique())}
        
        # Create the DataFrame with aligned arrays
        portfolio_df = pd.DataFrame({
            'ticker': tickers[:n_stocks],
            'weight': optimized_weights[:n_stocks]
        })
        
        # Add sector information
        if 'sector' in stock_data.columns:
            # Create ticker to sector mapping
            ticker_to_sector = dict(zip(sector_data['ticker'], sector_data['sector']))
            portfolio_df['sector'] = portfolio_df['ticker'].map(ticker_to_sector)
        else:
            # Use numeric sector IDs if sector names not available
            portfolio_df['sector_id'] = sectors_array[:n_stocks]
            portfolio_df['sector'] = portfolio_df['sector_id'].map(lambda x: sector_dict.get(x, f"Sector {x}"))
            portfolio_df = portfolio_df.drop('sector_id', axis=1)
        
        # Sort by weight descending (absolute value)
        portfolio_df['abs_weight'] = portfolio_df['weight'].abs()
        portfolio_df = portfolio_df.sort_values('abs_weight', ascending=False)
        portfolio_df = portfolio_df.drop('abs_weight', axis=1)
        
        # print("\nPortfolio DataFrame (Top 10 positions by weight):")
        # print(portfolio_df.head(10))
        
        
        # # Save the DataFrame to CSV
        # portfolio_df.to_csv('portfolio_weights.csv', index=False)
        
        return portfolio_df, problem.status

    return None
"""