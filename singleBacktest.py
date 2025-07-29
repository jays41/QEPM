import pandas as pd
from get_betas_and_cov_matrix import get_preweighting_data
from stratified_weights import get_stratified_weights
from fixed_expected_returns import get_expected_returns_ending
from screenByFactors import get_z_scores_dataframe

print('Starting backtest...')

stock_returns = pd.read_csv(r"QEPM\data\all_data.csv")
technical_factor_data = pd.read_csv(r"QEPM\data\technical_factors.csv") # date
economic_factor_data = pd.read_csv(r"QEPM\data\econ_data.csv") # PeriodDate
fundamental_factor_data = pd.read_csv(r"QEPM\data\stock_fundamental_data.csv") # public_date

stock_returns['date'] = pd.to_datetime(stock_returns['date'])
stock_returns['date'] = stock_returns['date'].dt.to_period('M')
stock_returns['returns'] = stock_returns.groupby('gvkey')['close'].pct_change()
stock_returns = stock_returns.groupby(['gvkey', 'date'])['returns'].mean().reset_index()

economic_factor_data = (
    economic_factor_data.dropna(subset=['Series_Value'])
    .groupby(['PeriodDate', 'EcoSeriesID'])['Series_Value']
    .first()
    .reset_index()
)

economic_factor_data.columns = economic_factor_data.columns.str.strip()
economic_factor_data = economic_factor_data.pivot(index='PeriodDate', columns='EcoSeriesID', values='Series_Value')
economic_factor_data = economic_factor_data.reset_index()

economic_factor_data['PeriodDate'] = pd.to_datetime(economic_factor_data['PeriodDate']).dt.to_period('M')

economic_factor_data = stock_returns.merge(economic_factor_data, left_on='date', right_on='PeriodDate', how='left')

economic_factor_data = (
    economic_factor_data.groupby(['date', 'gvkey'])
    .first()
    .reset_index()
)

fundamental_factor_data['public_date'] = pd.to_datetime(fundamental_factor_data['public_date']).dt.to_period('M')

columns_to_keep = ['gvkey', 'public_date', 'npm', 'opmad', 'gpm', 'ptpm', 'pretret_earnat', 'equity_invcap', 'debt_invcap', 'capital_ratio', 'invt_act', 'rect_act', 'debt_assets', 'debt_capital', 'cash_ratio', 'adv_sale']
fundamental_factor_data = fundamental_factor_data[columns_to_keep]

fundamental_factor_data = (
    fundamental_factor_data.groupby(['public_date', 'gvkey'])
    .mean()
    .reset_index()
)

fundamental_factor_data = stock_returns.merge(
    fundamental_factor_data,
    left_on=['date', 'gvkey'],
    right_on=['public_date', 'gvkey'],
    how='left'
).drop(columns=['public_date'])

technical_factor_data['date'] = pd.to_datetime(technical_factor_data['date']).dt.to_period('M')

technical_factor_data = (
    technical_factor_data
    .groupby(['gvkey', 'date'])[['macd_30']]
    .mean()
    .reset_index()
)

technical_factor_data = technical_factor_data.merge(
    fundamental_factor_data,  
    left_on=['gvkey', 'date'], 
    right_on=['gvkey', 'date'],  
    how='left'
)

print('Formatted data from csvs')

price_data = pd.read_csv(r"QEPM\data\all_data.csv")
price_data['date'] = pd.to_datetime(price_data['date'])

technical_factor_data['date'] = technical_factor_data['date'].dt.to_timestamp()
economic_factor_data['PeriodDate'] = economic_factor_data['PeriodDate'].dt.to_timestamp()
fundamental_factor_data['date'] = fundamental_factor_data['date'].dt.to_timestamp()

###########################################################################

def calculate_turnover(prev_weights, curr_weights):
    """Calculate portfolio turnover between two weight vectors"""
    if prev_weights is None or prev_weights.empty:
        return curr_weights.abs().sum()  # Full turnover on first investment
    
    # Align indices and fill missing with 0
    all_stocks = prev_weights.index.union(curr_weights.index)
    prev_aligned = prev_weights.reindex(all_stocks, fill_value=0)
    curr_aligned = curr_weights.reindex(all_stocks, fill_value=0)
    
    return (prev_aligned - curr_aligned).abs().sum()

def backtest(lookback_start_month, lookback_start_year, lookback_end_month, lookback_end_year, invest_start_month, invest_start_year, invest_end_month, invest_end_year, previous_weights=None):
    lookback_start = pd.Timestamp(f"{lookback_start_year}-{lookback_start_month}-01")
    lookback_end = pd.Timestamp(f"{lookback_end_year}-{lookback_end_month}-01") + pd.offsets.MonthEnd(0)
    end_date_for_expected_returns = f'{lookback_end_year}-{lookback_end_month}'
    
    invest_start = pd.Timestamp(f"{invest_start_year}-{invest_start_month}-01")
    invest_end = pd.Timestamp(f"{invest_end_year}-{invest_end_month}-01") + pd.offsets.MonthEnd(0)

    start_date = f"{lookback_start_year}-{lookback_start_month}-01"
    end_date = f"{lookback_end_year}-{lookback_end_month}-28"
    
    screened_stocks_df = get_z_scores_dataframe(
        start_date=start_date,
        end_date=end_date,
        winsorise_percentile=0.05,  # remove top/bottom 5% outliers
        top_percentile=0.2          # then select top/bottom 20%
    )
    
    if screened_stocks_df.empty:
        print(f"Warning: No stocks passed screening for period {start_date} to {end_date}")
        return 0, False, pd.Series()
    
    screened_stocks = screened_stocks_df['stock'].tolist()
    print(f"Selected {len(screened_stocks)} stocks from screening")
    
    expected_returns_df = get_expected_returns_ending(end_date_for_expected_returns)
    expected_returns_df = expected_returns_df.reset_index()
    
    # Filter expected returns to only include screened stocks
    expected_returns_df = expected_returns_df[expected_returns_df['gvkey'].isin(screened_stocks)]
    
    if expected_returns_df.empty:
        print(f"Warning: No expected returns data for screened stocks")
        return 0, False, pd.Series()

    stock_data, expected_returns, cov_matrix, betas, sectors = get_preweighting_data(
        expected_returns_df, lookback_start, lookback_end
    )
    
    # check if there is enough stocks for optimisation
    if len(stock_data) < 5:  # Minimum threshold
        print(f"Warning: Too few stocks ({len(stock_data)}) for optimization")
        return 0, False, pd.Series()

    target_annual_risk = 0.04
    portfolio_df, problem_status = get_stratified_weights(stock_data, expected_returns, cov_matrix, betas, sectors, target_annual_risk)
    print(f'Optimized weights for {len(portfolio_df)} screened stocks')

    price_data['date'] = pd.to_datetime(price_data['date'])
    actual_invest_start_date = price_data[price_data['date'] >= invest_start]['date'].min()
    actual_invest_end_date = price_data[price_data['date'] <= invest_end]['date'].max()
    
    if pd.isna(actual_invest_start_date) or pd.isna(actual_invest_end_date):
        print(f"Warning: Missing price data for investment period {invest_start} to {invest_end}")
        return 0, False, portfolio_df['weight'] if 'weight' in portfolio_df.columns else pd.Series()
    
    start_prices = price_data[price_data['date'] == actual_invest_start_date].set_index('gvkey')['close']
    end_prices = price_data[price_data['date'] == actual_invest_end_date].set_index('gvkey')['close']

    stock_prices = pd.read_csv(r"QEPM/data/stock_prices.csv")
    translation_table = stock_prices[['gvkey', 'ticker']].drop_duplicates()
    ticker_to_gvkey = translation_table.set_index('ticker')['gvkey'].to_dict()

    portfolio_df['gvkey'] = portfolio_df['ticker'].map(ticker_to_gvkey)
    portfolio_df = portfolio_df.dropna(subset=['gvkey'])
    portfolio_df['gvkey'] = portfolio_df['gvkey'].astype(int)
    portfolio_df = portfolio_df.set_index('gvkey')

    available_stocks = set(start_prices.index) & set(end_prices.index) & set(portfolio_df.index)
    missing_stocks = set(portfolio_df.index) - available_stocks
    
    if missing_stocks:
        print(f"Warning: {len(missing_stocks)} stocks missing price data, removing from portfolio")
        portfolio_df = portfolio_df.loc[list(available_stocks)]
        
        if not portfolio_df.empty:
            portfolio_df['weight'] = portfolio_df['weight'] / portfolio_df['weight'].sum()
        else:
            print("Error: No stocks with complete price data")
            return 0, False, pd.Series()

    returns = (end_prices / start_prices - 1).fillna(0)
    portfolio_return = (returns * portfolio_df['weight']).sum()
    
    transaction_cost_bps = 10  # 10 basis points per transaction
    current_weights = portfolio_df['weight']
    turnover = calculate_turnover(previous_weights, current_weights)
    transaction_cost = turnover * (transaction_cost_bps / 10000) # Convert bps to decimal
    
    # net_return = portfolio_return - transaction_cost
    net_return = portfolio_return # transaction cost of 0 for now
    
    print(f"Portfolio return from {invest_start} to {invest_end}: {(100 * portfolio_return):.6f}%")
    print(f"Transaction cost: {(100 * transaction_cost):.4f}%")
    print(f"Net return: {(100 * net_return):.6f}%")
    
    return net_return, problem_status == "optimal", current_weights