from get_5_year_preweighting_data import get_5_year_preweighting_data
from stratified_weights import get_stratified_weights
from backtest_1_year import backtest_1_year
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
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
# preprocess data -> get 5 year window
results = []
while end_year <= final_year:
    # get data from the period start_year to end_year
    data = stock_data[(stock_data['date'].dt.year >= start_year) & (stock_data['date'].dt.year <= end_year)]
    print(f'got data for range {start_year} to {end_year}')
    
    stock_data, expected_returns, cov_matrix, betas, sectors_array = get_5_year_preweighting_data(data)
    print(f'got preweighting data for range {start_year} to {end_year}')
    
    portfolio_df, status = get_stratified_weights(stock_data, expected_returns, cov_matrix, betas, sectors_array, target_annual_risk)
    print(f'got stratified weights for range {start_year} to {end_year}')

    ann_return = backtest_1_year(stock_data, portfolio_df, end_year + 1)
    results.append([end_year + 1, ann_return])
    print(f'With a target annual risk of {target_annual_risk}, annual return was {ann_return} for the year {end_year + 1}')
    
    start_year += 1
    end_year += 1

for r in results:
    print(f'Return for year {r[0]}: {ann_return}')