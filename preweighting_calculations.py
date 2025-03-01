import pandas as pd
import numpy as np
import yfinance as yf

def get_preweighting_data():
    # Assuming your data is loaded in a DataFrame called 'stock_data'
    # with columns ['ticker', 'date', 'close_price', 'sector']
    stock_data = pd.read_csv(r"QEPM\data\stock_prices.csv")
    # Convert date to datetime if it's not already
    stock_data['date'] = pd.to_datetime(stock_data['date'])

    # Sort by ticker and date
    stock_data = stock_data.sort_values(['ticker', 'date'])

    duplicate_check = stock_data.duplicated(subset=['ticker', 'date'], keep=False)
    if duplicate_check.any():
        print(f"Found {duplicate_check.sum()} duplicate entries. Taking the last entry for each ticker-date pair.")
        # Keep the last entry for each ticker-date pair
        stock_data = stock_data.drop_duplicates(subset=['ticker', 'date'], keep='last')

    # Calculate daily returns by ticker
    stock_data['return'] = stock_data.groupby('ticker')['close'].pct_change()

    # Drop rows with NaN returns (typically the first row for each ticker)
    stock_data = stock_data.dropna(subset=['return'])

    # Create a pivot table of returns with dates as index and tickers as columns
    returns_pivot = stock_data.pivot(index='date', columns='ticker', values='return')

    # Create a pivot table for sectors
    sector_mapping = stock_data.drop_duplicates('ticker').set_index('ticker')['sector']

    # Method 1: Historical mean returns (simplest approach)
    # expected_returns = returns_pivot.mean()

    # Method 2: More sophisticated approach - exponentially weighted moving average
    # This gives more weight to recent returns
    lookback_period = 252  # One trading year
    expected_returns_ewma = returns_pivot.ewm(span=lookback_period).mean().iloc[-1]


    # Method 1: Historical covariance
    # cov_matrix = returns_pivot.cov()

    # Method 2: Exponentially weighted covariance (more weight to recent data)
    # Create exponential weights (more weight to recent observations)
    recent_period = min(252, len(returns_pivot))
    recent_returns = returns_pivot.iloc[-recent_period:]

    # Check for NaN values that might be causing issues
    print(f"NaN values in recent returns: {recent_returns.isna().sum().sum()}")

    # Fill NaN values with 0 or method of your choice
    recent_returns = recent_returns.fillna(0)

    # Create weights and calculate covariance
    weights = np.exp(np.linspace(-1, 0, recent_period))
    weights = weights / weights.sum()

    # Adjust returns by their mean and apply weights
    weighted_returns = recent_returns.subtract(recent_returns.mean())
    weighted_returns = weighted_returns.multiply(np.sqrt(weights[:, np.newaxis]), axis=0)

    # Calculate weighted covariance the correct way
    cov_matrix = weighted_returns.T @ weighted_returns


    # Create or load market index returns
    # You can use a broad market index like S&P 500
    # Example: Load market data and calculate returns
    # market_data = pd.read_csv('market_index.csv')
    # market_data['return'] = market_data['close'].pct_change().dropna()

    # For illustration, let's assume we have market returns in a Series called market_returns
    # with the same date index as our stock returns

    # Download market index data using yfinance
    start_date = returns_pivot.index.min().strftime('%Y-%m-%d')
    end_date = returns_pivot.index.max().strftime('%Y-%m-%d')
    market_data = yf.download('^GSPC', start=start_date, end=end_date)  # ^GSPC is the S&P 500 ticker


    # Calculate market returns using the appropriate column
    # Try 'Close' if 'Adj Close' is not available
    if 'Adj Close' in market_data.columns:
        market_returns = market_data['Adj Close'].pct_change().dropna()
    else:
        market_returns = market_data['Close'].pct_change().dropna()

    # Ensure market_returns has matching dates with stock returns
    market_returns = market_returns[market_returns.index.isin(returns_pivot.index)]



    # Calculate Beta for each stock
    betas = {}
    for ticker in returns_pivot.columns:
        stock_returns = returns_pivot[ticker].dropna()
        # Align market returns with stock returns
        aligned_data = pd.concat([stock_returns, market_returns], axis=1).dropna()
        stock_return = aligned_data.iloc[:, 0]
        market_return = aligned_data.iloc[:, 1]
        
        # Calculate covariance and market variance
        cov = stock_return.cov(market_return)
        market_var = market_return.var()
        
        # Beta = Covariance(stock, market) / Variance(market)
        beta = cov / market_var
        betas[ticker] = beta

    # Convert to Series
    betas = pd.Series(betas)


    # Prepare data for your existing optimization code
    n = len(expected_returns_ewma)
    expected_returns_array = expected_returns_ewma.values
    cov_matrix_array = cov_matrix.values
    betas_array = betas.values

    # Map sectors to numeric values for use in [stratified_weights.py](QEPM/stratified_weights.py)
    sectors_array = sector_mapping.map({sector: i for i, sector in enumerate(sector_mapping.unique())}).values


    return expected_returns_array, cov_matrix_array, betas_array, sectors_array