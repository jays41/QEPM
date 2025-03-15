# market cap <- can add market cap for next year
# momentum -> calculate manually

import pandas as pd
from clean_data import get_fundamental_data

stock_data = pd.read_csv(r"QEPM\data\all_data.csv")
selected_data = stock_data[['gvkey', 'date', 'gind']] # currently doing by industry but can change
selected_data = selected_data.drop_duplicates()
selected_data.reset_index(drop=True, inplace=True)

fundamental_data = get_fundamental_data()


# stock_list is a list of gvkeys
stock_list = selected_data.drop_duplicates().gvkey.tolist()


factors = ['ticker', 'price-book', 'EV-EBITDA', 'free-cash-operating-cash', 'price-earnings-growth-dividend-yield', 'return-on-common-equity', 'interest-coverage-ratio', 'total-asset-turnover', 'dividend-yield']

# need a dataset mapping stocks to sectors
def get_sector(stock):
    # to implement
    pass

def get_z_scores(start_date, end_date, stock_list, factors):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    overall_z_scores = []
    number_of_stocks = len(stock_list)
    for i in range(len(stock_list)):
        stock, sector = stock_list.iloc[i]['gvkey'], stock_list.iloc[i]['gind']
        print(f'calculating z-score for stock {i} of {number_of_stocks}')
        z = 0
        for factor in factors:
            lastValue, mean, std = get_factor_data(start_date, end_date, stock, factor)
            z += (lastValue - mean) / std
        # TODO: add weighting here -> from Tristan's regressions
        overall_z_scores.append((z, stock, sector))
    overall_z_scores.sort(key=lambda x : x[0])
    n = len(overall_z_scores)
    #FIGURE THIS OUT
    top = overall_z_scores[:n//10]
    bottom = overall_z_scores[n - n//10 :] # change this to only take the bottom 10% of the positive ones
    return top + bottom


def get_factor_data(start_date, end_date, stock, factor):
    
    stock_factor_data = fundamental_data[(fundamental_data['gvkey'] == stock) & (fundamental_data[factor] == factor)]
    print(stock_factor_data.head())
    stock_factor_data = stock_factor_data[(stock_factor_data['date'] >= start_date) & (stock_factor_data['date'] <= end_date)]
    print(stock_data.head())
    last_year_price_data = stock_data[stock_data['date'] >= (pd.to_datetime(end_date) - pd.DateOffset(years=1))]
    mean = last_year_price_data['close'].mean()
    std = last_year_price_data['close'].std()
    lastValue = last_year_price_data['close'].iloc[-1]
    
    return lastValue, mean, std
    
    # get data for the factor for the year
    # filter by stock
    # calculate mean and std for the last year
    # return lastValue, mean, std
    
# MAIN
start_date = "2010-01-04"
end_date = "2023-12-12"
get_z_scores(start_date, end_date, selected_data, factors)