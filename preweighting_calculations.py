import pandas as pd
import numpy as np
import yfinance as yf

# Need to fix cov matrices
# Need to include factors in expected returns

def get_preweighting_data():
    # Load stock data
    stock_data = pd.read_csv(r"QEPM\data\stock_prices.csv")
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    stock_data = stock_data.sort_values(['ticker', 'date'])

    # Remove duplicates
    duplicate_check = stock_data.duplicated(subset=['ticker', 'date'], keep=False)
    if duplicate_check.any():
        # print(f"Found {duplicate_check.sum()} duplicate entries. Taking the last entry for each ticker-date pair.")
        stock_data = stock_data.drop_duplicates(subset=['ticker', 'date'], keep='last')

    # Calculate daily returns
    stock_data['return'] = stock_data.groupby('ticker')['close'].pct_change()
    
    # Drop rows with NaN returns and filter out extreme outliers
    stock_data = stock_data.dropna(subset=['return'])
    
    # Cap extreme returns (e.g., due to stock splits, M&A)
    upper_limit = stock_data['return'].quantile(0.99) * 1.5
    lower_limit = stock_data['return'].quantile(0.01) * 1.5
    # print(f"Capping extreme returns: below {lower_limit:.4f} and above {upper_limit:.4f}")
    stock_data.loc[stock_data['return'] > upper_limit, 'return'] = upper_limit
    stock_data.loc[stock_data['return'] < lower_limit, 'return'] = lower_limit

    # Create pivot table of returns
    returns_pivot = stock_data.pivot(index='date', columns='ticker', values='return')
    
    # Create sector mapping
    sector_mapping = stock_data.drop_duplicates('ticker').set_index('ticker')['sector']
    
    # Calculate expected returns using EWMA
    lookback_period = 252  # One trading year
    # print(f"Calculating expected returns with EWMA using {lookback_period} day lookback")
    expected_returns_ewma = returns_pivot.ewm(span=lookback_period).mean().iloc[-1]
    
    # Validate expected returns
    if expected_returns_ewma.isna().any():
        # print(f"Warning: {expected_returns_ewma.isna().sum()} NaN values in expected returns")
        expected_returns_ewma = expected_returns_ewma.fillna(expected_returns_ewma.median())
    
    # Cap extreme expected returns
    returns_cap = 0.01  # Cap daily expected returns at ±1%
    expected_returns_ewma = expected_returns_ewma.clip(-returns_cap, returns_cap)
    # print(f"Expected returns range: {expected_returns_ewma.min():.4f} to {expected_returns_ewma.max():.4f}")

    # Calculate covariance matrix using recent data
    recent_period = min(252, len(returns_pivot))
    recent_returns = returns_pivot.iloc[-recent_period:]
    # print(f"Calculating covariance using {recent_period} most recent days")
    
    # Remove stocks with too many missing values
    missing_threshold = 0.3  # 30%
    missing_pct = recent_returns.isna().mean()
    tickers_to_keep = missing_pct[missing_pct < missing_threshold].index
    if len(tickers_to_keep) < len(recent_returns.columns):
        # print(f"Removing {len(recent_returns.columns) - len(tickers_to_keep)} tickers with >30% missing data")
        recent_returns = recent_returns[tickers_to_keep]
        expected_returns_ewma = expected_returns_ewma[tickers_to_keep]
        sector_mapping = sector_mapping[sector_mapping.index.isin(tickers_to_keep)]
    
    # Fill remaining NaN values
    print(f"Filling {recent_returns.isna().sum().sum()} NaN values in returns data")
    recent_returns = recent_returns.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # Compute weighted covariance matrix
    weights = np.exp(np.linspace(-1, 0, recent_period))
    weights = weights / weights.sum()
    demeaned_returns = recent_returns.subtract(recent_returns.mean())
    weighted_returns = demeaned_returns.multiply(np.sqrt(weights[:, np.newaxis]), axis=0)
    cov_matrix = weighted_returns.T @ weighted_returns
    
    # Ensure covariance matrix is well-conditioned
    min_eigenvalue = np.min(np.linalg.eigvals(cov_matrix))
    if min_eigenvalue < 1e-6:
        # print(f"Adding regularization to covariance matrix (min eigenvalue: {min_eigenvalue:.8f})")
        cov_matrix = cov_matrix + np.eye(len(cov_matrix)) * max(1e-6, 1e-4 * np.trace(cov_matrix)/len(cov_matrix))
    
    # Calculate betas - with improved handling of market data
    try:
        # Download market index data
        start_date = returns_pivot.index.min().strftime('%Y-%m-%d')
        end_date = returns_pivot.index.max().strftime('%Y-%m-%d')
        # print(f"Downloading market data from {start_date} to {end_date}")
        market_data = yf.download('^GSPC', start=start_date, end=end_date)
        
        # Calculate market returns
        market_returns = market_data['Adj Close'].pct_change().dropna()
        
        # Align market index with stock returns
        aligned_market = market_returns.reindex(recent_returns.index)
        if aligned_market.isna().sum() > 0:
            print(f"Warning: {aligned_market.isna().sum()} missing market data points. Filling with forward fill.")
            aligned_market = aligned_market.fillna(method='ffill').fillna(0)
        
        # Calculate betas using vectorized operations
        common_dates = recent_returns.index.intersection(market_returns.index)
        if len(common_dates) < 0.7 * recent_period:
            print(f"Warning: Only {len(common_dates)} common dates for beta calculation (expected {recent_period})")
        
        # Calculate betas for each stock
        betas = {}
        market_var = aligned_market.var()
        if market_var <= 0:
            print("Warning: Market variance is zero or negative. Using placeholder beta values.")
            betas = pd.Series(1.0, index=recent_returns.columns)
        else:
            for ticker in recent_returns.columns:
                stock_return = recent_returns[ticker]
                cov_with_market = stock_return.cov(aligned_market)
                beta = cov_with_market / market_var
                # Cap extreme beta values
                betas[ticker] = np.clip(beta, -3.0, 3.0)
            betas = pd.Series(betas)
        
        # Check for NaN or infinite betas
        if betas.isna().any() or np.isinf(betas).any():
            print(f"Warning: {betas.isna().sum()} NaN and {np.isinf(betas).sum()} infinite beta values")
            betas = betas.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        
        # print(f"Beta range: {betas.min():.4f} to {betas.max():.4f}")
    
    except Exception as e:
        print(f"Error calculating betas: {e}")
        print("Using placeholder beta values of 1.0")
        betas = pd.Series(1.0, index=recent_returns.columns)

    # Prepare final data arrays
    expected_returns_array = expected_returns_ewma.values
    cov_matrix_array = cov_matrix.values  
    betas_array = betas.values
    
    # Map sectors to numeric values
    sector_dict = {sector: i for i, sector in enumerate(sector_mapping.unique())}
    sectors_array = sector_mapping.map(sector_dict).values
    
    # Final validation
    # print(f"\nFinal data preparation complete:")
    # print(f"- Assets: {len(expected_returns_array)}")
    # print(f"- Expected returns: {expected_returns_array.min():.4f} to {expected_returns_array.max():.4f}")
    # print(f"- Covariance matrix shape: {cov_matrix_array.shape}")
    # print(f"- Beta values: {betas_array.min():.4f} to {betas_array.max():.4f}")
    # print(f"- Unique sectors: {len(np.unique(sectors_array))}")



    return stock_data, expected_returns_array, cov_matrix_array, betas_array, sectors_array