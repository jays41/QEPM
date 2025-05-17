import pandas as pd
import numpy as np
import yfinance as yf

def get_preweighting_data(expected_returns_df: pd.DataFrame):
    """
    Prepares data for portfolio optimization.

    Parameters:
        expected_returns_df (pd.DataFrame): DataFrame with columns ['gvkey', 'Expected Return'].

    Returns:
        stock_data (pd.DataFrame): Original stock data with returns and adjustments.
        expected_returns (pd.Series): Expected returns indexed by gvkey.
        cov_matrix (pd.DataFrame): Covariance matrix of returns.
        betas (pd.Series): CAPM betas indexed by gvkey.
        sectors (pd.Series): Sector mapping indexed by gvkey.
    """
    
    
     # Debugging: Check the structure of expected_returns_df
    print("Columns in expected_returns_df:", expected_returns_df.columns)

    # Ensure 'gvkey' column exists
    if 'gvkey' not in expected_returns_df.columns:
        raise KeyError("The 'expected_returns_df' DataFrame must contain a 'gvkey' column.")

    # Set 'gvkey' as the index
    expected_returns_df = expected_returns_df.set_index('gvkey')

    # Ensure 'Expected Return' column exists
    if 'Expected Return' not in expected_returns_df.columns:
        raise KeyError("The 'expected_returns_df' DataFrame must contain an 'Expected Return' column.")

    expected_returns = expected_returns_df['Expected Return']
    
    # Load stock data
    stock_data = pd.read_csv(r"QEPM\data\stock_prices.csv")
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    stock_data = stock_data.sort_values(['ticker', 'date'])

    # Remove duplicates
    stock_data = stock_data.drop_duplicates(subset=['ticker', 'date'], keep='last')

    # Calculate daily returns
    stock_data['return'] = stock_data.groupby('ticker')['close'].pct_change()

    # Drop rows with NaN returns and filter out extreme outliers
    stock_data = stock_data.dropna(subset=['return'])
    upper_limit = stock_data['return'].quantile(0.99) * 1.5
    lower_limit = stock_data['return'].quantile(0.01) * 1.5
    stock_data.loc[stock_data['return'] > upper_limit, 'return'] = upper_limit
    stock_data.loc[stock_data['return'] < lower_limit, 'return'] = lower_limit

    # Create pivot table of returns
    returns_pivot = stock_data.pivot(index='date', columns='ticker', values='return')

    # Create sector mapping
    sector_mapping = stock_data.drop_duplicates('ticker').set_index('ticker')['sector']

    # Map gvkey to tickers
    gvkey_mapping = stock_data.drop_duplicates('ticker').set_index('ticker')['gvkey']

    # Validate expected returns
    
    # expected_returns_df = expected_returns_df.set_index('gvkey')
    # expected_returns = expected_returns_df['Expected Return']
    tickers_to_keep = gvkey_mapping[gvkey_mapping.isin(expected_returns.index)].index

    # Filter returns and sector mapping to match expected returns
    returns_pivot = returns_pivot[tickers_to_keep]
    sector_mapping = sector_mapping[sector_mapping.index.isin(tickers_to_keep)]
    gvkey_mapping = gvkey_mapping[gvkey_mapping.index.isin(tickers_to_keep)]

    # Calculate covariance matrix using recent data
    recent_period = min(252, len(returns_pivot))
    recent_returns = returns_pivot.iloc[-recent_period:]

    # Remove stocks with too many missing values
    missing_threshold = 0.3  # 30%
    missing_pct = recent_returns.isna().mean()
    tickers_to_keep = missing_pct[missing_pct < missing_threshold].index
    recent_returns = recent_returns[tickers_to_keep]
    sector_mapping = sector_mapping[sector_mapping.index.isin(tickers_to_keep)]
    gvkey_mapping = gvkey_mapping[gvkey_mapping.index.isin(tickers_to_keep)]
    expected_returns = expected_returns[expected_returns.index.isin(gvkey_mapping.values)]

    # Fill remaining NaN values
    recent_returns = recent_returns.ffill().bfill().fillna(0)

    # Compute weighted covariance matrix
    weights = np.exp(np.linspace(-1, 0, recent_period))
    weights = weights / weights.sum()
    demeaned_returns = recent_returns.subtract(recent_returns.mean())
    weighted_returns = demeaned_returns.multiply(np.sqrt(weights[:, np.newaxis]), axis=0)
    cov_matrix = weighted_returns.T @ weighted_returns
    cov_matrix = pd.DataFrame(cov_matrix, index=recent_returns.columns, columns=recent_returns.columns)

    # Ensure covariance matrix is well-conditioned
    min_eigenvalue = np.min(np.linalg.eigvals(cov_matrix))
    if min_eigenvalue < 1e-6:
        cov_matrix += np.eye(len(cov_matrix)) * max(1e-6, 1e-4 * np.trace(cov_matrix) / len(cov_matrix))

    # Calculate betas
    try:
        start_date = returns_pivot.index.min().strftime('%Y-%m-%d')
        end_date = returns_pivot.index.max().strftime('%Y-%m-%d')
        market_data = yf.download('^GSPC', start=start_date, end=end_date, auto_adjust=False)
        market_returns = market_data['Adj Close'].pct_change().dropna()
        aligned_market = market_returns.reindex(recent_returns.index).fillna(0)

        betas = {}
        market_var = aligned_market.var()
        for ticker in recent_returns.columns:
            stock_return = recent_returns[ticker]
            cov_with_market = stock_return.cov(aligned_market)
            beta = cov_with_market / market_var
            betas[ticker] = np.clip(beta, -3.0, 3.0)
        betas = pd.Series(betas)
    except Exception as e:
        print(f"Error calculating betas: {e}")
        betas = pd.Series(1.0, index=recent_returns.columns)

    # Map betas to gvkey
    betas.index = gvkey_mapping[betas.index]
    betas = betas.values

    # Map sectors to numeric values
    sector_dict = {sector: i for i, sector in enumerate(sector_mapping.unique())}
    sectors = sector_mapping.map(sector_dict)

    return stock_data, expected_returns, cov_matrix, betas, sectors