import pandas as pd
import numpy as np

def get_preweighting_data(
    expected_returns_df: pd.DataFrame,
    start_date,
    end_date,
    returns_freq: str = 'M',
    invest_start=None,
    invest_end=None,
):

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
    
    # check tickers that were there on start_date (set) and remove any others
    stock_data = stock_data[(stock_data['date'] >= start_date) & (stock_data['date'] <= end_date)]
    # get min date
    min_date = stock_data['date'].min()
    # find unique gvkeys on that date
    gvkeys_on_min_date = set(stock_data[stock_data['date'] == min_date]['gvkey'])
    # keep only those in contention
    stock_data = stock_data[stock_data['gvkey'].isin(gvkeys_on_min_date)]

    # Calculate returns at requested frequency
    if returns_freq == 'Q':
        stock_data['Period'] = stock_data['date'].dt.to_period('Q')
    else:
        stock_data['Period'] = stock_data['date'].dt.to_period('M')
    agg = (stock_data
           .groupby(['ticker', 'Period'])
           .agg({'close': 'last', 'gvkey': 'last', 'sector': 'last'})
           .reset_index())
    agg['date'] = agg['Period'].dt.to_timestamp(how='end')
    agg['return'] = agg.groupby('ticker')['close'].pct_change()
    stock_data = agg

    # Drop rows with NaN returns and filter out extreme outliers
    stock_data = stock_data.dropna(subset=['return'])
    upper_limit = stock_data['return'].quantile(0.99) * 1.5
    lower_limit = stock_data['return'].quantile(0.01) * 1.5
    stock_data.loc[stock_data['return'] > upper_limit, 'return'] = upper_limit
    stock_data.loc[stock_data['return'] < lower_limit, 'return'] = lower_limit

    # Create pivot table of returns (ticker columns)
    returns_pivot = stock_data.pivot(index='date', columns='ticker', values='return')

    # Create sector mapping and gvkey mapping (ticker-indexed)
    meta = stock_data.drop_duplicates('ticker')[['ticker', 'gvkey', 'sector']].set_index('ticker')
    sector_mapping = meta['sector']
    gvkey_mapping = meta['gvkey']

    # Validate expected returns
    
    # expected_returns_df = expected_returns_df.set_index('gvkey')
    # expected_returns = expected_returns_df['Expected Return']
    tickers_to_keep = gvkey_mapping[gvkey_mapping.isin(expected_returns.index)].index

    # Filter returns and sector mapping to match expected returns universe
    returns_pivot = returns_pivot[tickers_to_keep]
    sector_mapping = sector_mapping.loc[tickers_to_keep]
    gvkey_mapping = gvkey_mapping.loc[tickers_to_keep]

    # Rename columns to gvkey and deduplicate (prefer last if multiple tickers map to same gvkey)
    returns_pivot.columns = returns_pivot.columns.map(gvkey_mapping)
    if returns_pivot.columns.duplicated().any():
        returns_pivot = returns_pivot.groupby(level=0, axis=1).last()
    # Build sector mapping by gvkey
    sectors_df = pd.DataFrame({'gvkey': gvkey_mapping.values, 'sector': sector_mapping.values})
    sectors_gv = sectors_df.drop_duplicates('gvkey').set_index('gvkey')['sector']

    # Align expected_returns to available gvkeys and preserve order
    universe = expected_returns.index.intersection(returns_pivot.columns)
    expected_returns = expected_returns.loc[universe]
    returns_pivot = returns_pivot[universe]
    sectors_gv = sectors_gv.reindex(universe)

    # Enforce invest-window tradability: require non-NaN returns for all periods in invest window
    if invest_start is not None and invest_end is not None:
        invest_mask = (returns_pivot.index >= pd.to_datetime(invest_start)) & (
            returns_pivot.index <= pd.to_datetime(invest_end)
        )
        if invest_mask.any():
            tradable = returns_pivot.loc[invest_mask].notna().all(axis=0)
            tradable_universe = tradable[tradable].index
            # Re-align to tradable subset
            expected_returns = expected_returns.loc[tradable_universe]
            returns_pivot = returns_pivot[tradable_universe]
            sectors_gv = sectors_gv.reindex(tradable_universe)

    # Calculate covariance matrix using recent data
    horizon = 36 if returns_freq == 'M' else 12  # window used to slice recent history
    recent_period = min(horizon, len(returns_pivot))
    recent_returns = returns_pivot.iloc[-recent_period:]

    # Remove stocks with excessive missing values in recent window
    missing_threshold = 0.5  # allow up to 50% missing in recent returns window
    missing_pct = recent_returns.isna().mean()
    keep_cols = missing_pct[missing_pct <= missing_threshold].index
    recent_returns = recent_returns[keep_cols]
    expected_returns = expected_returns.reindex(keep_cols).dropna()
    sectors_gv = sectors_gv.reindex(expected_returns.index)
    recent_returns = recent_returns[expected_returns.index]

    # Pairwise covariance with constant-correlation shrinkage
    min_history_periods = 24 if returns_freq == 'M' else 8
    emp_cov = recent_returns.cov(min_periods=min_history_periods)
    # If emp_cov is empty or 1x1, fall back to diagonal with sample variances
    if emp_cov.shape[0] <= 1:
        variances = recent_returns.var().replace([np.inf, -np.inf], np.nan).fillna(1e-4)
        cov_matrix = np.diag(variances.values)
        cov_matrix = pd.DataFrame(cov_matrix, index=recent_returns.columns, columns=recent_returns.columns)
    else:
        # Replace NaNs with 0 on diagonal and compute std, clamp tiny stds
        emp_cov_filled = emp_cov.fillna(0)
        diag = np.diag(emp_cov_filled.values)
        diag = np.where(diag <= 0, 1e-8, diag)
        std = np.sqrt(diag)
        std = np.where(std <= 1e-8, 1e-8, std)
        # Compute correlation matrix
        with np.errstate(invalid='ignore', divide='ignore'):
            denom = np.outer(std, std)
            corr = emp_cov_filled.values / denom
        # Set diagonal to 1
        np.fill_diagonal(corr, 1.0)
        # Average off-diagonal correlation; if all NaN, set to 0.0
        off_mask = ~np.eye(corr.shape[0], dtype=bool)
        off_vals = corr[off_mask]
        if np.isnan(off_vals).all():
            rho_bar = 0.0
            corr = np.where(np.isnan(corr), 0.0, corr)
        else:
            rho_bar = np.nanmean(off_vals)
            corr = np.where(np.isnan(corr), rho_bar, corr)
        # Shrinkage
        lam = 0.3
        J = np.ones_like(corr)
        target_corr = rho_bar * (J - np.eye(corr.shape[0])) + np.eye(corr.shape[0])
        shrunk_corr = (1 - lam) * corr + lam * target_corr
        # Reconstruct covariance
        cov_matrix = (shrunk_corr * denom)
        cov_matrix = pd.DataFrame(cov_matrix, index=recent_returns.columns, columns=recent_returns.columns)
        # Replace any residual NaNs or infs
        cov_matrix = cov_matrix.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Ensure positive definiteness by jitter if needed
    try:
        np.linalg.cholesky(cov_matrix.values)
    except np.linalg.LinAlgError:
        trace = np.trace(cov_matrix.values)
        jitter = 1e-6 * (trace / max(1, len(cov_matrix))) if trace > 0 else 1e-6
        cov_matrix.values[range(len(cov_matrix)), range(len(cov_matrix))] += jitter

    # Reorder stock_data to match expected_returns order (gvkey) for consistent ticker mapping
    # Build gvkey -> ticker map (choose last occurrence if duplicates)
    gv_to_ticker = meta.reset_index().drop_duplicates('gvkey', keep='last').set_index('gvkey')['ticker']
    tickers_in_order = [gv_to_ticker.get(gv) for gv in expected_returns.index]
    tickers_in_order = [t for t in tickers_in_order if t is not None]
    if 'ticker' in stock_data.columns:
        stock_data = stock_data[stock_data['ticker'].isin(tickers_in_order)].copy()
        stock_data['ticker'] = pd.Categorical(stock_data['ticker'], categories=tickers_in_order, ordered=True)
        stock_data = stock_data.sort_values('ticker')

    # Calculate betas
    try:
        start_date = returns_pivot.index.min().strftime('%Y-%m-%d')
        end_date = returns_pivot.index.max().strftime('%Y-%m-%d')
        market_data = pd.read_csv(r"QEPM\data\s&p_data.csv")
        # Normalize column names for robustness
        market_data.columns = [c.strip() for c in market_data.columns]
        date_col = 'Date' if 'Date' in market_data.columns else ('date' if 'date' in market_data.columns else None)
        adj_col = 'Adj Close' if 'Adj Close' in market_data.columns else ('AdjClose' if 'AdjClose' in market_data.columns else None)
        if date_col is None or adj_col is None:
            raise KeyError("s&p_data.csv must contain 'Date' (or 'date') and 'Adj Close' (or 'AdjClose') columns")
        market_data[date_col] = pd.to_datetime(market_data[date_col])
        market_data = market_data[(market_data[date_col] >= pd.to_datetime(start_date)) & (market_data[date_col] <= pd.to_datetime(end_date))]
        market_data = market_data.set_index(date_col)
        market_data[adj_col] = pd.to_numeric(market_data[adj_col], errors='coerce')
        if returns_freq == 'Q':
            market_data['Quarter'] = market_data.index.to_period('Q')
            market_series = market_data.groupby('Quarter')[adj_col].last().to_timestamp(how='end')
            market_returns = market_series.pct_change().dropna()
        else:
            market_data['Month'] = market_data.index.to_period('M')
            market_series = market_data.groupby('Month')[adj_col].last().to_timestamp(how='end')
            market_returns = market_series.pct_change().dropna()
        aligned_market = market_returns.reindex(recent_returns.index).fillna(0)

        betas = {}
        market_var = aligned_market.var()
        for gv in recent_returns.columns:
            stock_return = recent_returns[gv]
            cov_with_market = stock_return.cov(aligned_market)
            beta = cov_with_market / market_var
            betas[gv] = np.clip(beta, -3.0, 3.0)
        betas = pd.Series(betas)
    except Exception as e:
        print(f"Error calculating betas: {e}")
        betas = pd.Series(1.0, index=recent_returns.columns)

    # Ensure outputs aligned to expected_returns index (gvkey order)
    betas = betas.reindex(expected_returns.index).fillna(1.0).values

    # Map sectors to numeric values in gvkey order
    sector_dict = {sector: i for i, sector in enumerate(sectors_gv.dropna().unique())}
    sectors = sectors_gv.map(sector_dict).reindex(expected_returns.index).fillna(-1).astype(int).values
    
    # print(f"betas:\n{betas}")
    
    return stock_data, expected_returns, cov_matrix, betas, sectors