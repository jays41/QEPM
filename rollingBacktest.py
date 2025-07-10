import pandas as pd
from get_betas_and_cov_matrix import get_preweighting_data
from stratified_weights import get_stratified_weights
from fixed_expected_returns import get_expected_returns_ending

print('Starting backtest...')

# Ensure stock_returns have prct_change
stock_returns = pd.read_csv(r"QEPM\data\all_data.csv")
technical_factor_data = pd.read_csv(r"QEPM\data\technical_factors.csv") # date
economic_factor_data = pd.read_csv(r"QEPM\data\econ_data.csv") # PeriodDate
fundamental_factor_data = pd.read_csv(r"QEPM\data\stock_fundamental_data.csv") # public_date

# Convert daily stock returns to monthly
stock_returns['date'] = pd.to_datetime(stock_returns['date'])
stock_returns['date'] = stock_returns['date'].dt.to_period('M')
stock_returns['returns'] = stock_returns.groupby('gvkey')['close'].pct_change()
stock_returns = stock_returns.groupby(['gvkey', 'date'])['returns'].mean().reset_index()

# ECONOMIC FACTOR DATA MANIPULATION
economic_factor_data = (
    economic_factor_data.dropna(subset=['Series_Value'])  # Drops rows where Series_Value is NaN
    .groupby(['PeriodDate', 'EcoSeriesID'])['Series_Value']
    .mean()  # You can use sum(), first(), last(), etc.
    .reset_index()
)

# Pivot the economic data to wide format
economic_factor_data.columns = economic_factor_data.columns.str.strip()
economic_factor_data = economic_factor_data.pivot(index='PeriodDate', columns='EcoSeriesID', values='Series_Value')
economic_factor_data = economic_factor_data.reset_index()

# Merge stock returns with economic_factor_data using one-to-many on date
economic_factor_data['PeriodDate'] = pd.to_datetime(economic_factor_data['PeriodDate']).dt.to_period('M')

economic_factor_data = (
    economic_factor_data.groupby(['PeriodDate'])
    .mean()  # You can use sum(), first(), last(), etc.
    .reset_index()
)

economic_factor_data = stock_returns.merge(economic_factor_data, left_on='date', right_on='PeriodDate', how='left')

economic_factor_data = (
    economic_factor_data.groupby(['PeriodDate', 'gvkey'])
    .mean()  # You can use sum(), first(), last(), etc.
    .reset_index()
)

# Fill missing data for fundamental factors using forward and backward fill
# fundamental_factor_data = fundamental_factor_data.fillna(method='ffill').fillna(method='bfill')

# Convert 'public_date' to datetime format (ensuring consistency)
fundamental_factor_data['public_date'] = pd.to_datetime(fundamental_factor_data['public_date']).dt.to_period('M')

# Select relevant columns
columns_to_keep = ['gvkey', 'public_date', 'npm', 'opmad', 'gpm', 'ptpm', 'pretret_earnat', 'equity_invcap', 'debt_invcap', 'capital_ratio', 'invt_act', 'rect_act', 'debt_assets', 'debt_capital', 'cash_ratio', 'adv_sale']
fundamental_factor_data = fundamental_factor_data[columns_to_keep]

# Aggregate fundamental data by 'gvkey' and 'public_date'
fundamental_factor_data = (
    fundamental_factor_data.groupby(['public_date', 'gvkey'])
    .mean()  # You can replace with sum(), first(), etc.
    .reset_index()
)

# Merge stock_returns with fundamental factors
fundamental_factor_data = stock_returns.merge(
    fundamental_factor_data,
    left_on=['date', 'gvkey'],
    right_on=['public_date', 'gvkey'],
    how='left'
).drop(columns=['public_date'])  # Drop duplicate column

# Convert 'date' to Period('M') format
technical_factor_data['date'] = pd.to_datetime(technical_factor_data['date']).dt.to_period('M')

# Group by 'gvkey' and 'date' while taking the mean of technical factors
technical_factor_data = (
    technical_factor_data
    .groupby(['gvkey', 'date'])[['macd_30']]
    .mean()
    .reset_index()
)

# Merge technical factors with fundamental factors correctly
technical_factor_data = technical_factor_data.merge(
    fundamental_factor_data,  
    left_on=['gvkey', 'date'], 
    right_on=['gvkey', 'date'],  
    how='left'
)

print('Formatted data from csvs')




# get betas
# get Z-scores and screen
price_data = pd.read_csv(r"QEPM\data\all_data.csv")
price_data['date'] = pd.to_datetime(price_data['date'])
screened_stocks = pd.read_csv(r"QEPM\data\z_scores_all_betas.csv")
screened_stocks = screened_stocks['stock']
print(screened_stocks)

print('Retrieved screened stocks')

# Ensure 'date' in technical_factor_data is in datetime format
technical_factor_data['date'] = technical_factor_data['date'].dt.to_timestamp()
# Ensure 'PeriodDate' in economic_factor_data is in datetime format
economic_factor_data['PeriodDate'] = economic_factor_data['PeriodDate'].dt.to_timestamp()
# Ensure 'public_date' in fundamental_factor_data is in datetime format
fundamental_factor_data['date'] = fundamental_factor_data['date'].dt.to_timestamp()


###########################################################################


def backtest(lookback_start_month, lookback_start_year, lookback_end_month, lookback_end_year, invest_start_month, invest_start_year, invest_end_month, invest_end_year):
    lookback_start = pd.Timestamp(f"{lookback_start_year}-{lookback_start_month}-01")
    lookback_end = pd.Timestamp(f"{lookback_end_year}-{lookback_end_month}-01") + pd.offsets.MonthEnd(0)
    end_date_for_expected_returns = f'{lookback_end_year}-{lookback_end_month}'
    
    invest_start = pd.Timestamp(f"{invest_start_year}-{invest_start_month}-01")
    invest_end = pd.Timestamp(f"{invest_end_year}-{invest_end_month}-01") + pd.offsets.MonthEnd(0)

    
    # get betas and cov matrices
    expected_returns_df = get_expected_returns_ending(end_date_for_expected_returns) # CHECK THAT IT ONLY DOES lookback_period NUMBER OF MONTHS BEFORE
    expected_returns_df = expected_returns_df.reset_index()

    stock_data, expected_returns, cov_matrix, betas, sectors = get_preweighting_data(expected_returns_df, lookback_start, lookback_end) # CHECK FORMAT OF THE BETAS
    # print('Got betas, covariance matrices')

    # optimise: stratified_weights.py
    # print('Optimising...')
    target_annual_risk = 0.05
    portfolio_df, problem_status = get_stratified_weights(stock_data, expected_returns, cov_matrix, betas, sectors, target_annual_risk)
    # print(portfolio_df)
    print('Optimised weights')


    price_data['date'] = pd.to_datetime(price_data['date'])
    actual_invest_start_date = price_data[price_data['date'] >= invest_start]['date'].min()
    # Find the last trading date on or before invest_end_date
    actual_invest_end_date = price_data[price_data['date'] <= invest_end]['date'].max()
    start_prices = price_data[price_data['date'] == actual_invest_start_date].set_index('gvkey')['close']
    end_prices = price_data[price_data['date'] == actual_invest_end_date].set_index('gvkey')['close']

    stock_prices = pd.read_csv(r"QEPM/data/stock_prices.csv")

    translation_table = stock_prices[['gvkey', 'ticker']].drop_duplicates()
    ticker_to_gvkey = translation_table.set_index('ticker')['gvkey'].to_dict()

    portfolio_df['gvkey'] = portfolio_df['ticker'].map(ticker_to_gvkey)

    portfolio_df = portfolio_df.dropna(subset=['gvkey'])
    portfolio_df['gvkey'] = portfolio_df['gvkey'].astype(int)

    portfolio_df = portfolio_df.set_index('gvkey')

    missing_start = set(portfolio_df.index) - set(start_prices.index)
    missing_end = set(portfolio_df.index) - set(end_prices.index)
    # print("Missing in start_prices:", missing_start)
    # print("Missing in end_prices:", missing_end)

    price_change = end_prices - start_prices
    portfolio_df['price_change'] = price_change
    portfolio_df['weighted_return'] = portfolio_df['price_change'] * portfolio_df['weight']
    portfolio_return = portfolio_df['weighted_return'].sum()
    print(f"Portfolio return from {invest_start} to {invest_end}: {(100 * portfolio_return):.6f}%")
    
    return portfolio_return, problem_status == "optimal"