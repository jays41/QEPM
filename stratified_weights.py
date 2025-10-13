import pandas as pd
import numpy as np
import cvxpy as cp

def get_stratified_weights(stock_data, expected_returns, cov_matrix, betas, sectors_array, target_annual_risk, periods_per_year=12):
    # Sanitize inputs (dtype only here)
    expected_returns = np.asarray(expected_returns, dtype=float)
    betas = np.asarray(betas, dtype=float)
    sectors_array = np.asarray(sectors_array)
    cov_matrix = np.asarray(cov_matrix, dtype=float)

    # Helper: impute NaNs/Infs in ER with sector medians then global median
    def _impute_er(er: np.ndarray, sectors: np.ndarray) -> np.ndarray:
        er = er.copy()
        # Treat infs as NaN for imputation
        er[~np.isfinite(er)] = np.nan
        if np.any(np.isnan(er)):
            s = pd.Series(sectors)
            er_s = pd.Series(er)
            # Sector median map (dropna per sector)
            med_map = er_s.groupby(s).median()
            # Fill per sector where possible
            for sec, med in med_map.items():
                if np.isfinite(med):
                    mask = (s.values == sec) & np.isnan(er)
                    er[mask] = med
            # Global median fallback
            if np.any(np.isnan(er)):
                global_med = np.nanmedian(er)
                if not np.isfinite(global_med):
                    global_med = 0.0
                er[np.isnan(er)] = global_med
        return er

    # Helper: impute NaNs/Infs in betas with sector medians then default 1.0, clamp
    def _impute_betas(b: np.ndarray, sectors: np.ndarray) -> np.ndarray:
        b = b.copy()
        b[~np.isfinite(b)] = np.nan
        if np.any(np.isnan(b)):
            s = pd.Series(sectors)
            b_s = pd.Series(b)
            med_map = b_s.groupby(s).median()
            for sec, med in med_map.items():
                if np.isfinite(med):
                    mask = (s.values == sec) & np.isnan(b)
                    b[mask] = med
            if np.any(np.isnan(b)):
                b[np.isnan(b)] = 1.0
        # Clamp extreme betas
        b = np.clip(b, -3.0, 3.0)
        return b

    # Helper: sanitize covariance by imputing correlations and ensuring PD
    def _sanitize_cov(cov: np.ndarray) -> np.ndarray:
        cov = 0.5 * (cov + cov.T)
        cov = cov.copy()
        # Replace inf with NaN for processing
        cov[~np.isfinite(cov)] = np.nan
        # Variances
        var = np.diag(cov)
        # If any diag NaN/<=0, set small positive
        var = np.where(~np.isfinite(var) | (var <= 0), 1e-6, var)
        std = np.sqrt(var)
        # Build corr with NaNs where denom invalid
        denom = np.outer(std, std)
        with np.errstate(invalid='ignore', divide='ignore'):
            corr = cov / denom
        # Set diag to 1
        np.fill_diagonal(corr, 1.0)
        # Compute average off-diagonal correlation ignoring NaNs
        off_mask = ~np.eye(corr.shape[0], dtype=bool)
        off_vals = corr[off_mask]
        if np.isnan(off_vals).all():
            rho_bar = 0.0
        else:
            rho_bar = np.nanmean(off_vals)
        # Impute NaNs in corr with rho_bar
        corr = np.where(np.isnan(corr), rho_bar, corr)
        # Reconstruct covariance
        cov_rec = corr * denom
        cov_rec = 0.5 * (cov_rec + cov_rec.T)
        # Ensure PD with jitter
        try:
            np.linalg.cholesky(cov_rec)
        except np.linalg.LinAlgError:
            trace = np.trace(cov_rec)
            jitter = 1e-6 * (trace / max(1, cov_rec.shape[0])) if trace > 0 else 1e-6
            cov_rec = cov_rec + np.eye(cov_rec.shape[0]) * jitter
        return cov_rec

    # Apply imputations
    expected_returns = _impute_er(expected_returns, sectors_array)
    betas = _impute_betas(betas, sectors_array)
    cov_matrix = _sanitize_cov(cov_matrix)
    # Force symmetry one more time
    cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)
    n = len(expected_returns)

    # Convert annual volatility target to per-period volatility (risk constraint uses stdev units)
    target_risk = target_annual_risk / np.sqrt(periods_per_year)

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

    # print("\nSector Distribution:")
    for sector_id in unique_sectors:
        stock_count = len(sector_indices[sector_id])
        proportion = sector_proportions[sector_id] * 100
        # print(f"Sector {sector_id}: {stock_count} stocks ({proportion:.2f}%)")


    w = cp.Variable(n)  # Stock weights

    # Ensure PD via jitter if needed
    try:
        cov_chol = np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        trace = np.trace(cov_matrix)
        jitter = 1e-6 * (trace / max(1, n)) if trace > 0 else 1e-6
        cov_matrix = cov_matrix + np.eye(n) * jitter
        cov_chol = np.linalg.cholesky(cov_matrix)

    # Objective Function: Maximize Expected Return
    # Center ER to reduce degeneracy; still maximize total expected return
    er_centered = expected_returns - np.median(expected_returns)
    objective = cp.Maximize(cp.sum(cp.multiply(er_centered, w)))

    # Constraints
    constraints = []
    # Dollar neutrality with scalar tolerance
    constraints.append(cp.abs(cp.sum(w)) <= 1e-3)
    # Risk constraint (SOC)
    constraints.append(cp.norm(cov_chol @ w, 2) <= target_risk)
    # Gross exposure and box bounds
    constraints.append(cp.norm1(w) <= 2)
    constraints += [w >= -0.2, w <= 0.2]

    # Adaptive sector constraints
    tiny_universe = n < 10
    strict_min = 4   # >= 4 names → exact neutrality (unless tiny universe)
    soft_min = 2     # 2–3 names → soft neutrality
    tol_small = 0.02
    tol_tiny = 0.05
    for sector_id in unique_sectors:
        indices = sector_indices[sector_id]
        k = len(indices)
        if k >= strict_min and not tiny_universe:
            # Exact neutrality only when enough names and universe isn't tiny
            constraints.append(cp.sum(w[indices]) == 0)
        elif k >= soft_min:
            # Soft neutrality tolerance for small sectors
            tol = tol_tiny if tiny_universe else tol_small
            constraints.append(cp.abs(cp.sum(w[indices])) <= tol)
        # else: k == 1 → skip neutrality
        # Proportional sector exposure with a floor
        max_sector_exposure = max(0.25, sector_proportions[sector_id] * 2.2)
        constraints.append(cp.sum(cp.abs(w[indices])) <= max_sector_exposure)

    # Beta neutrality with tolerance
    beta_tol = 0.05 if tiny_universe else 0.03
    constraints.append(cp.abs(cp.sum(cp.multiply(betas, w))) <= beta_tol)


    # Try solving the problem with improved solver parameters
    try:
        # Try SCS with better parameters
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS, eps=1e-8, max_iters=10000000000, alpha=1.8)
        # print("Solved with SCS using improved parameters")
    except Exception as e:
        # print(f"SCS solver failed: {e}")
        try:
            # Try OSQP if SCS fails
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8, max_iter=10000)
            # print("Solved with OSQP")
        except Exception as e:
            # print(f"OSQP solver failed: {e}")
            try:
                # Try ECOS if OSQP fails
                problem = cp.Problem(objective, constraints)
                problem.solve(solver=cp.ECOS, abstol=1e-8, reltol=1e-8, max_iters=1000)
                # print("Solved with ECOS")
            except Exception as e:
                # print(f"ECOS solver failed: {e}")
                print("All standard solvers failed. Trying alternative approach.")
                
                # Final attempt with CVXOPT with better parameters
                problem = cp.Problem(objective, constraints)
                problem.solve(solver=cp.CVXOPT, abstol=1e-8, reltol=1e-8)
                # print("Solved with CVXOPT using improved parameters")

    # Output optimized weights
    if problem.status == "optimal" or problem.status == "optimal_inaccurate":
        optimized_weights = w.value
        if optimized_weights is None:
            optimized_weights = np.zeros(n)
        optimized_weights = np.nan_to_num(optimized_weights, nan=0.0, posinf=0.0, neginf=0.0)
        print("\nOptimization successful!")
        print("Status:", problem.status)
        # print("Optimized weights:", optimized_weights)
        
        # Calculate the long and short positions
        long_positions = np.sum(optimized_weights[optimized_weights > 0])
        short_positions = np.sum(optimized_weights[optimized_weights < 0])
        
        # Calculate actual risk using original covariance matrix
        portfolio_variance = optimized_weights @ cov_matrix @ optimized_weights
        portfolio_volatility = np.sqrt(portfolio_variance)

        # Calculating per-period expected return
        expected_period_return = np.sum(expected_returns * optimized_weights)

        print("expected_period_return:", expected_period_return)
        print("expected_returns (min, max):", np.min(expected_returns), np.max(expected_returns))
        print("optimized_weights (min, max):", np.min(optimized_weights), np.max(optimized_weights))
        if expected_period_return < -1 or expected_period_return > 1:
            print("Warning: expected_daily_returns is outside typical range!")

        annual_return = (1 + expected_period_return) ** periods_per_year - 1
        annual_volatility = portfolio_volatility * np.sqrt(periods_per_year)
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

    # Fallback: return empty portfolio if unsolved
    return None