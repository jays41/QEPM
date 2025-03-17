# Need the betas to be able to correctly calculate the Z-scores

# To do:
# add econ factors to Z-score calculation
# use the betas for the factors rather than the raw values

from collections import defaultdict
import pandas as pd
from clean_data import get_fundamental_data

stock_data = pd.read_csv(r"QEPM\data\all_data.csv")
selected_data = stock_data[['gvkey', 'date', 'gind']] # currently doing by industry but can change
selected_data.reset_index(drop=True, inplace=True)
selected_data = selected_data.drop_duplicates()

fundamental_data = get_fundamental_data()

d = defaultdict(int)

# stock_list is a list of gvkeys
stock_list = selected_data.drop_duplicates(subset=['gvkey'])


# 15 econ factors chosen
econ_factors = [200380, 592, 598, 2177, 134896, 202661, 202664, 202811, 202813, 201723, 137439, 148429, 202074, 202600, 202605]
econ_factor_data = pd.read_csv(r"QEPM\data\econ_data.csv")
econ_factor_data = econ_factor_data[['EcoSeriesID', 'PeriodDate', 'Series_Value']]
econ_factor_data.columns = ['factor_key', 'date', 'value']

# 26 fundamental factors TO CHOOSE
factors = ['dividend-yield', 'EV-EBITDA', 'price-book', 'price-cf', 'price-earnings', 'price-EBITDA', 'price-sales', 'price-earnings-growth', 'price-earnings-growth-dividend-yield', 'cash-ratio', 'current-ratio', 'quick-ratio', 'inventory-turnover', 'receivables-turnover', 'total-asset-turnover', 'cash-conversion-cycle', 'gross-profit-margin', 'net-profit-margin', 'operating-profit-margin', 'return-on-assets', 'return-on-common-equity', 'return-on-total-capital', 'debt-equity', 'total-debt-ratio', 'interest-coverage-ratio', 'free-cash-operating-cash']

def get_z_scores(start_date, end_date, stock_list, factors):
    complete = 0
    incomplete = 0
    total = 0
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    overall_z_scores = []
    number_of_stocks = len(stock_list)
    econ_factor_data = econ_factor_data[(econ_factor_data['date'] >= start_date) & (econ_factor_data['date'] <= end_date)]
    for i in range(len(stock_list)):
        stock, sector = stock_list.iloc[i]['gvkey'], stock_list.iloc[i]['gind']
        # print(f'calculating z-score for stock {i} of {number_of_stocks}')
        z = 0
        number_of_factors_used = 0
        for fundamental_factor in factors:
            lastValue, mean, std = get_fundamental_factor_data(start_date, end_date, stock, fundamental_factor)
            if lastValue != 0 or mean != 0 or std != 1:
                number_of_factors_used += 1
            # print(f'for factor: {factor} -> lastValue={lastValue}, mean={mean}, std={std}')
            z += (lastValue - mean) / std
        # TODO: add weighting here -> from Tristan's regressions
        for econ_factor in econ_factors:
            lastValue, mean, std = get_econ_factor_data()
        overall_z_scores.append((z, stock, sector))
        if number_of_factors_used == 6:
            complete += 1
        else:
            incomplete += 1
        total += len(factors) - number_of_factors_used
        print(f'z-score for {stock} is {z}')
    overall_z_scores.sort(key=lambda x : x[0])
    n = len(overall_z_scores)
    overall_z_scores = [x for x in overall_z_scores if x[0] > 0]
    top = overall_z_scores[:n//10]
    bottom = overall_z_scores[len(overall_z_scores) - n//10 :]
    print(top)
    print('---------')
    print(bottom)
    print(f'n: {n}')
    print(f'average = total missing / n : {total/n}')
    print(f'complete: {complete}')
    print(f'incomplete: {incomplete}')
    return top + bottom


def get_fundamental_factor_data(start_date, end_date, stock, factor):
    global d
    stock_factor_data = fundamental_data[fundamental_data['gvkey'] == stock]
    stock_factor_data = stock_factor_data[['gvkey', 'public-date', factor]]
    stock_factor_data = stock_factor_data[(stock_factor_data['public-date'] >= start_date) & (stock_factor_data['public-date'] <= end_date)]
    
    # Check if the factor column contains string values
    if stock_factor_data[factor].dtype == 'object':
        stock_factor_data[factor] = stock_factor_data[factor].str.rstrip('%').astype('float')
    
    if stock_factor_data.empty or stock_factor_data[factor].isna().any():
        d[factor] += 1
        return 0, 0, 1
    
    mean = stock_factor_data[factor].mean()
    std = stock_factor_data[factor].std()
    lastValue = stock_factor_data[factor].iloc[-1]
    
    return lastValue, mean, std
    
    # get data for the factor for the year
    # filter by stock
    # calculate mean and std for the last year
    # return lastValue, mean, std

def get_econ_factor_data(start_date, end_date, stock, factor):
    pass
    
# MAIN
start_date = "2010-01-04"
end_date = "2011-12-12"
for i in range(2011, 2023, 1):
    end_date = f'{i}-12-12'
    get_z_scores(start_date, end_date, stock_list, factors)
    
for k, v in d:
    print(f'factor: {k}, missing {v} times')
