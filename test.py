from get_5_year_preweighting_data import get_5_year_preweighting_data
import pandas as pd

'''
make it rolling - do expected returns based on a 5 year look back period and weight and then run the backtest on the upcoming year
and then move the whole window by one year and do the same thing over and over again
'''

target_annual_risk = 0.05

stock_data = pd.read_csv(r"QEPM\data\stock_prices.csv")
stock_data['date'] = pd.to_datetime(stock_data['date'])

start_year = 2010
end_year = 2014
final_year = 2021

data = stock_data[(stock_data['date'].dt.year >= start_year) & (stock_data['date'].dt.year <= end_year)]
print(f'got data for range {start_year} to {end_year}')
print(data)    
expected_returns, cov_matrix, betas, sectors_array = get_5_year_preweighting_data(data)
print(data)
print(expected_returns)
print(cov_matrix)
print(betas)
print(sectors_array)