import pandas as pd
from get_betas_and_cov_matrix import get_preweighting_data
from stratified_weights import get_stratified_weights
from testing_expected_returns import get_expected_returns


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
screened_stocks = pd.read_csv(r"QEPM\data\z_scores_all_betas.csv")
screened_stocks = screened_stocks['stock']
print(screened_stocks)

print('Retrieved screened stocks')


###########################################################################



start_month = '12'
end_year = '2020'
start_year = str(int(end_year)-3)
end_month = start_month
start_date = pd.Timestamp(f"{start_year}-{start_month}-01")
end_date = pd.Timestamp(f"{end_year}-{end_month}-01") + pd.offsets.MonthEnd(0)
end_date_for_expected_returns = f'{end_year}-{start_month}'


# get sam's expected return
# filter the data according to the year
# Create a filtered version of technical_factor_data
# Ensure 'date' in technical_factor_data is in datetime format
technical_factor_data['date'] = technical_factor_data['date'].dt.to_timestamp()
# Ensure 'PeriodDate' in economic_factor_data is in datetime format
economic_factor_data['PeriodDate'] = economic_factor_data['PeriodDate'].dt.to_timestamp()
# Ensure 'public_date' in fundamental_factor_data is in datetime format
fundamental_factor_data['date'] = fundamental_factor_data['date'].dt.to_timestamp()
# Filter the data for the specified date range
# technical_factor_data_to_use = technical_factor_data[(technical_factor_data['date'] >= start_date) & (technical_factor_data['date'] <= end_date)]
# economic_factor_data_to_use = economic_factor_data[(economic_factor_data['PeriodDate'] >= start_date) & (economic_factor_data['PeriodDate'] <= end_date)]
# fundamental_factor_data_to_use = fundamental_factor_data[(fundamental_factor_data['date'] >= start_date) & (fundamental_factor_data['date'] <= end_date)]

# expected_returns_df, tau_values, av_momentum = get_expected_returns(economic_factor_data, fundamental_factor_data, technical_factor_data, end_date_for_expected_returns)
# print('Calculated expected returns')

# get betas and cov matrices
expected_returns_df = pd.read_csv(r"QEPM\exret\returns_ending_2020_12.csv")

stock_data, expected_returns, cov_matrix, betas, sectors = get_preweighting_data(expected_returns_df) # CHECK FORMAT OF THE BETAS
print('Got betas, covariance matrices')

# optimise: stratified_weights.py
print('Optimising...')
target_annual_risk = 0.05
portfolio_df, problem_status = get_stratified_weights(stock_data, expected_returns, cov_matrix, betas, sectors, target_annual_risk)
print(portfolio_df)
print('Optimised weights')
portfolio_df.to_csv(r"QEPM\results\results_2020_12.csv", index=False)
# calculate returns from raw price change multiplied by weight and sum for all stocks in the portfolio

start_date = price_data[(price_data['date'].dt.year == start_year) & (price_data['date'].dt.month == start_month)]['date'].min()
end_date = price_data[(price_data['date'].dt.year == end_year) & (price_data['date'].dt.month == end_month)]['date'].max()

# for each ticker
price_data['date'] = pd.to_datetime(price_data['date'])
start_prices = price_data[price_data['date'] == start_date].set_index('gvkey')['close']
end_prices = price_data[price_data['date'] == end_date].set_index('gvkey')['close']

portfolio_df['gvkey'] = portfolio_df['ticker'].astype(int)  # Assuming ticker is gvkey
portfolio_df = portfolio_df.set_index('gvkey')
price_change = end_prices - start_prices
portfolio_df['price_change'] = price_change
portfolio_df['weighted_return'] = portfolio_df['price_change'] * portfolio_df['weight']
portfolio_return = portfolio_df['weighted_return'].sum()
print(f"Portfolio return from {start_date} to {end_date}: {portfolio_return:.6f}")
# move the window up by one quarter