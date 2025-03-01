import numpy as np
import cvxpy as cp
from preweighting_calculations import get_preweighting_data

expected_returns, cov_matrix, betas, sectors_array = get_preweighting_data()

n = len(expected_returns)


target_risk = 0.1

# Small tolerance for equality constraints
epsilon = 1e-6



unique_sectors = np.unique(sectors_array)
num_sectors = len(unique_sectors)

# Calculate sector proportions and indices for each sector
sector_proportions = {}
sector_indices = {}
for sector_id in unique_sectors:
    # Get indices of stocks belonging to this sector
    indices = np.where(sectors_array == sector_id)[0]
    sector_indices[sector_id] = indices
    
    sector_proportions[sector_id] = len(indices) / n # Calculate proportion of stocks in this sector

print("\nSector Distribution:")
for sector_id in unique_sectors:
    stock_count = len(sector_indices[sector_id])
    proportion = sector_proportions[sector_id] * 100
    print(f"Sector {sector_id}: {stock_count} stocks ({proportion:.2f}%)")


w = cp.Variable(n)  # Stock weights

cov_chol = np.linalg.cholesky(cov_matrix)

# Objective Function: Maximize Expected Return
objective = cp.Maximize(cp.sum(cp.multiply(expected_returns, w)))

# Constraints
constraints = [
    cp.norm(cp.sum(w), 2) <= epsilon,  # Dollar neutrality with tiny slack
    cp.norm(cp.sum(cp.multiply(betas, w)), 2) <= epsilon,  # Beta neutrality with slack
    cp.norm(cov_chol @ w, 2) <= target_risk,  # SOC formulation of risk constraint
    cp.norm(w, 1) <= 2,  # Gross exposure limit (100% long, 100% short)
    w >= -0.1,  # Short position limit
    w <= 0.1  # Long position limit
]

# Add sector-specific constraints
for sector_id in unique_sectors:
    indices = sector_indices[sector_id]
    sector_weight = w[indices]
    
    # Sector neutrality - sum of weights within sector should be zero
    constraints.append(cp.sum(sector_weight) == 0)
    
    # Proportional sector weights - gross exposure should be proportional to sector size
    max_sector_exposure = sector_proportions[sector_id] * 2  # Proportional to sector size
    constraints.append(cp.sum(cp.abs(sector_weight)) <= max_sector_exposure)


# Try solving the problem with improved solver parameters
try:
    # Try SCS with better parameters
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, eps=1e-8, max_iters=25000, alpha=1.8)
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
    print("Optimized weights:", optimized_weights)
    
    # Calculate the long and short positions
    long_positions = np.sum(optimized_weights[optimized_weights > 0])
    short_positions = np.sum(optimized_weights[optimized_weights < 0])
    
    # Calculate actual risk using original covariance matrix
    portfolio_variance = optimized_weights @ cov_matrix @ optimized_weights
    portfolio_volatility = np.sqrt(portfolio_variance)
    
    # Verification
    print("\nVerification:")
    print(f"Sum of weights (dollar neutrality, should be 0): {np.sum(optimized_weights):.8f}")
    print(f"Long positions: {long_positions:.6f}")
    print(f"Short positions: {short_positions:.6f}")
    print(f"Beta neutrality (should be 0): {np.sum(betas * optimized_weights):.8f}")
    print(f"Expected return: {np.sum(expected_returns * optimized_weights):.6f}")
    print(f"Portfolio variance: {portfolio_variance:.6f}")
    print(f"Portfolio volatility: {portfolio_volatility:.6f}")
    print(f"Target volatility: {target_risk:.6f}")
    print(f"Gross exposure: {np.sum(np.abs(optimized_weights)):.6f}")
    
    # Sector-wise analysis
    print("\nSector-wise Analysis:")
    for sector_id in unique_sectors:
        indices = sector_indices[sector_id]
        sector_weights = optimized_weights[indices]
        sector_long_positions = np.sum(sector_weights[sector_weights > 0])
        sector_short_positions = np.sum(sector_weights[sector_weights < 0])
        sector_exposure = np.sum(np.abs(sector_weights))
        max_allowed = sector_proportions[sector_id] * 2
        
        print(f"\nSector {sector_id}:")
        print(f"  Number of stocks: {len(indices)}")
        print(f"  Sum of weights (should be 0): {np.sum(sector_weights):.8f}")
        print(f"  Long positions: {sector_long_positions:.6f}")
        print(f"  Short positions: {sector_short_positions:.6f}")
        print(f"  Gross exposure: {sector_exposure:.6f} (max allowed: {max_allowed:.6f})")
        print(f"  Exposure constraint satisfied: {sector_exposure <= max_allowed + 1e-6}")

else:
    print("Optimization failed with status:", problem.status)

# Additional analysis of the portfolio
if problem.status == "optimal" or problem.status == "optimal_inaccurate":
    # Count positions
    num_long = np.sum(optimized_weights > 0.001)  # Positions > 0.1%
    num_short = np.sum(optimized_weights < -0.001)  # Positions < -0.1%
    num_neutral = np.sum(np.abs(optimized_weights) <= 0.001)  # Near-zero positions
    
    print("\nPortfolio Structure:")
    print(f"Number of long positions (>0.1%): {num_long}")
    print(f"Number of short positions (<-0.1%): {num_short}")
    print(f"Number of near-zero positions: {num_neutral}")
    
    # Calculate dollar-neutrality deviation
    print(f"Dollar-neutrality deviation: {np.abs(long_positions + short_positions):.8f}")
    
    # Calculate correlation with market factors
    if np.std(betas @ optimized_weights) > 0:
        market_correlation = np.corrcoef(betas, optimized_weights)[0,1]
        print(f"Correlation with market factor: {market_correlation:.8f}")
    
    # Print constraint satisfaction levels
    print("\nConstraint Verification:")
    print(f"Dollar neutrality error: {np.abs(np.sum(optimized_weights)):.8f}")
    print(f"Beta neutrality error: {np.abs(np.sum(betas * optimized_weights)):.8f}")
    print(f"Risk constraint: {portfolio_volatility:.6f} <= {target_risk:.6f} is {portfolio_volatility <= target_risk + 1e-6}")
    print(f"Gross exposure: {np.sum(np.abs(optimized_weights)):.6f} <= 2 is {np.sum(np.abs(optimized_weights)) <= 2 + 1e-6}")
    print(f"Position limits satisfied: {np.all(optimized_weights >= -0.1-1e-6) and np.all(optimized_weights <= 0.1+1e-6)}")