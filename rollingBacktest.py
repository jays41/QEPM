import pandas as pd
from get_betas_and_cov_matrix import get_preweighting_data
from stratified_weights import get_stratified_weights
from testing_expected_returns import get_expected_returns_ending

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

    # get sam's expected return
    # filter the data according to the year
    # Create a filtered version of technical_factor_data
    # Filter the data for the specified date range
    # technical_factor_data_to_use = technical_factor_data[(technical_factor_data['date'] >= start_date) & (technical_factor_data['date'] <= end_date)]
    # economic_factor_data_to_use = economic_factor_data[(economic_factor_data['PeriodDate'] >= start_date) & (economic_factor_data['PeriodDate'] <= end_date)]
    # fundamental_factor_data_to_use = fundamental_factor_data[(fundamental_factor_data['date'] >= start_date) & (fundamental_factor_data['date'] <= end_date)]

    # expected_returns_df, tau_values, av_momentum = get_expected_returns(economic_factor_data, fundamental_factor_data, technical_factor_data, end_date_for_expected_returns)
    # print('Calculated expected returns')

    # get betas and cov matrices
    ##### expected_returns_df = pd.read_csv(r"QEPM\exret\returns_ending_2020_12.csv")
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
    # portfolio_df.to_csv(r"QEPM\results\results_2020_12.csv", index=False)
    # calculate returns from raw price change multiplied by weight and sum for all stocks in the portfolio

    # start_date = price_data[
    #     (price_data['date'].dt.year == int(start_year)) & (price_data['date'].dt.month == int(start_month))
    # ]['date'].min()
    # end_date = price_data[
    #     (price_data['date'].dt.year == int(end_year)) & (price_data['date'].dt.month == int(end_month))
    # ]['date'].max()

    # for each ticker
    price_data['date'] = pd.to_datetime(price_data['date'])
    actual_invest_start_date = price_data[price_data['date'] >= invest_start]['date'].min()
    # Find the last trading date on or before invest_end_date
    actual_invest_end_date = price_data[price_data['date'] <= invest_end]['date'].max()
    start_prices = price_data[price_data['date'] == actual_invest_start_date].set_index('gvkey')['close']
    end_prices = price_data[price_data['date'] == actual_invest_end_date].set_index('gvkey')['close']

    # portfolio_df['gvkey'] = portfolio_df['ticker'].astype(int)  # Assuming ticker is gvkey
    stock_prices = pd.read_csv(r"QEPM/data/stock_prices.csv")

    # Create a unique gvkey-ticker mapping
    translation_table = stock_prices[['gvkey', 'ticker']].drop_duplicates()

    # If you want a dictionary for mapping ticker -> gvkey:
    ticker_to_gvkey = translation_table.set_index('ticker')['gvkey'].to_dict()

    # 2. Map tickers in portfolio_df to gvkey
    portfolio_df['gvkey'] = portfolio_df['ticker'].map(ticker_to_gvkey)

    # 3. Drop any rows where gvkey could not be mapped (to avoid NaN indices)
    portfolio_df = portfolio_df.dropna(subset=['gvkey'])
    portfolio_df['gvkey'] = portfolio_df['gvkey'].astype(int)

    # 4. Set gvkey as the index for alignment
    portfolio_df = portfolio_df.set_index('gvkey')

    missing_start = set(portfolio_df.index) - set(start_prices.index)
    missing_end = set(portfolio_df.index) - set(end_prices.index)
    # print("Missing in start_prices:", missing_start)
    # print("Missing in end_prices:", missing_end)

    # portfolio_df = portfolio_df.set_index('gvkey')
    price_change = end_prices - start_prices
    portfolio_df['price_change'] = price_change
    portfolio_df['weighted_return'] = portfolio_df['price_change'] * portfolio_df['weight']
    portfolio_return = portfolio_df['weighted_return'].sum()
    print(f"Portfolio return from {invest_start} to {invest_end}: {(100 * portfolio_return):.6f}%")
    
    return portfolio_return, problem_status == "optimal"






# need to tackle issue of gvkeys being delisted:
#   add rebalancing
#   calculate price immediately after rebalancing

# filter the dates in get_preweighting_data() which should help with the issue above






# MAIN

# backtest('01', '2019', '12', '2019', '01', '2020', '12', '2020')  # 1-year lookback, 1-year invest

investment = 100

# print("2020 Investments:")
# res = backtest('01', '2019', '12', '2019', '01', '2020', '03', '2020')  # 1-year lookback, Q1 invest
# investment = investment * (1 + res)
# res = backtest('04', '2019', '03', '2020', '04', '2020', '06', '2020')  # 1-year lookback, Q2 invest
# investment = investment * (1 + res)
# res = backtest('07', '2019', '06', '2020', '07', '2020', '09', '2020')  # 1-year lookback, Q3 invest
# investment = investment * (1 + res)
# res = backtest('10', '2019', '09', '2020', '10', '2020', '12', '2020')  # 1-year lookback, Q4 invest
# investment = investment * (1 + res)
# print(f"Portfolio value = {investment}")
# print(f"Profit = {investment - 100}")

# print("2021 Investments:")
# res = backtest('01', '2020', '12', '2020', '01', '2021', '03', '2021')  # 1-year lookback, Q1 invest
# investment = investment * (1 + res)
# res = backtest('04', '2020', '03', '2021', '04', '2021', '06', '2021')  # 1-year lookback, Q2 invest
# investment = investment * (1 + res)
# res = backtest('07', '2020', '06', '2021', '07', '2021', '09', '2021')  # 1-year lookback, Q3 invest
# investment = investment * (1 + res)
# res = backtest('10', '2020', '09', '2021', '10', '2021', '12', '2021')  # 1-year lookback, Q4 invest
# investment = investment * (1 + res)
# print(f"Portfolio value = {investment}")
# print(f"Profit since start of 2020 = {investment - 100}")

investment_values = []
revival_indices = []

for start_year, end_year in [('2015','2016'), ('2016','2017'), ('2017','2018'), ('2018','2019'), ('2019','2020'), ('2020','2021'), ('2021', '2022'), ('2022', '2023')]:
    print(f"{end_year} Investments:")
    res, isOptimal = backtest('01', start_year, '12', start_year, '01', end_year, '03', end_year)  # 1-year lookback, Q1 invest
    if isOptimal:
        investment = investment * (1 + res)
        if investment <= 0:
            investment = 100
            revival_indices.append(len(investment_values))
    investment_values.append((f"03-{end_year}", investment))
    res, isOptimal = backtest('04', start_year, '03', end_year, '04', end_year, '06', end_year)  # 1-year lookback, Q2 invest
    if isOptimal:
        investment = investment * (1 + res)
        if investment <= 0:
            investment = 100
            revival_indices.append(len(investment_values))
    investment_values.append((f"06-{end_year}", investment))
    res, isOptimal = backtest('07', start_year, '06', end_year, '07', end_year, '09', end_year)  # 1-year lookback, Q3 invest
    if isOptimal:
        investment = investment * (1 + res)
        if investment <= 0:
            investment = 100
            revival_indices.append(len(investment_values))
    investment_values.append((f"09-{end_year}", investment))
    res, isOptimal = backtest('10', start_year, '09', end_year, '10', end_year, '12', end_year)  # 1-year lookback, Q4 invest
    if isOptimal:
        investment = investment * (1 + res)
        if investment <= 0:
            investment = 100
            revival_indices.append(len(investment_values))
    investment_values.append((f"12-{end_year}", investment))
    print(f"Portfolio value = {investment}")
    print(f"Profit since start of 2016 = {investment - 100}")

print(investment_values)

import matplotlib.pyplot as plt

# Unzip the investment_values into two lists: dates and values
dates, values = zip(*investment_values)

plt.figure(figsize=(12, 6))
plt.plot(dates, values, marker='o')
plt.title('Portfolio Value Over Time')
plt.xlabel('Quarter-End')
plt.ylabel('Portfolio Value')
plt.xticks(rotation=45)
plt.grid(True)

# Add vertical lines at revival points
for idx in revival_indices:
    plt.axvline(x=dates[idx], color='red', linestyle='--', alpha=0.7, label='Revival' if idx == revival_indices[0] else "")

# Only show one legend entry for 'Revival'
handles, labels = plt.gca().get_legend_handles_labels()
if 'Revival' in labels:
    plt.legend()

plt.tight_layout()
plt.show()

# print("2023 Investments:")
# res = backtest('01', '2022', '12', '2022', '01', '2023', '03', '2023')  # 1-year lookback, Q1 invest
# investment = investment * (1 + res)
# res = backtest('04', '2022', '03', '2023', '04', '2023', '06', '2023')  # 1-year lookback, Q2 invest
# investment = investment * (1 + res)
# res = backtest('07', '2022', '06', '2023', '07', '2023', '09', '2023')  # 1-year lookback, Q3 invest
# investment = investment * (1 + res)
# res = backtest('10', '2022', '09', '2023', '10', '2023', '12', '2023')  # 1-year lookback, Q4 invest
# investment = investment * (1 + res)
# print(f"Portfolio value = {investment}")
# print(f"Profit since start of 2020 = {investment - 100}")

# backtest('10', '2018', '09', '2019', '10', '2019', '09', '2020')  # 1-year lookback, 1-year invest starting in Oct