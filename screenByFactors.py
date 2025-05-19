# Need the betas to be able to correctly calculate the Z-scores

# To do:
# add econ factors to Z-score calculation
# use the betas for the factors rather than the raw values

from collections import defaultdict
import pandas as pd
from clean_data import get_fundamental_data
from expected_returns import expected_returns_setup, get_betas

stock_returns, economic_factor_data, raw_fundamental_factor_data, technical_factor_data = expected_returns_setup()

fundamental_betas, skipped_fundamental_stocks = get_betas(raw_fundamental_factor_data, 'date')

stock_data = pd.read_csv(r"QEPM\data\all_data.csv")
selected_data = stock_data[['gvkey', 'date', 'gind']] # currently doing by industry but can change
selected_data.reset_index(drop=True, inplace=True)
selected_data = selected_data.drop_duplicates()

# fundamental_data = get_fundamental_data()

d = defaultdict(int)

# stock_list is a list of gvkeys
stock_list = selected_data.drop_duplicates(subset=['gvkey'])


# 15 econ factors chosen
econ_factors = [200380, 592, 598, 2177, 134896, 202661, 202664, 202811, 202813, 201723, 137439, 148429, 202074, 202600, 202605]
# econ_factor_data = pd.read_csv(r"QEPM\data\econ_beta_results.csv")
# econ_factor_data = econ_factor_data[['EcoSeriesID', 'PeriodDate', 'Series_Value']]
# econ_factor_data.columns = ['factor_key', 'date', 'value']

# 26 fundamental factors TO CHOOSE
# factors = ['dividend-yield', 'EV-EBITDA', 'price-book', 'price-cf', 'price-earnings', 'price-EBITDA', 'price-sales', 'price-earnings-growth', 'price-earnings-growth-dividend-yield', 'cash-ratio', 'current-ratio', 'quick-ratio', 'inventory-turnover', 'receivables-turnover', 'total-asset-turnover', 'cash-conversion-cycle', 'gross-profit-margin', 'net-profit-margin', 'operating-profit-margin', 'return-on-assets', 'return-on-common-equity', 'return-on-total-capital', 'debt-equity', 'total-debt-ratio', 'interest-coverage-ratio', 'free-cash-operating-cash']
factors = ['npm', 'opmad', 'gpm', 'ptpm', 'pretret_earnat', 'equity_invcap', 'debt_invcap', 'capital_ratio', 'invt_act', 'rect_act', 'debt_assets', 'debt_capital', 'cash_ratio', 'adv_sale']

def get_z_scores(start_date, end_date, stock_list, factors):
    complete = 0
    incomplete = 0
    total = 0
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    overall_z_scores = []
    number_of_stocks = len(stock_list)
    # econ_factor_data = econ_factor_data[(econ_factor_data['date'] >= start_date) & (econ_factor_data['date'] <= end_date)]
    for i in range(len(stock_list)):
        # print(f'finding z-score for stock number {i}')
        stock, sector = stock_list.iloc[i]['gvkey'], stock_list.iloc[i]['gind']
        # print(f'calculating z-score for stock {i} of {number_of_stocks}')
        z = 0
        for fundamental_factor in factors:
            factorsUsed = 0
            # lastValue, mean, std = get_fundamental_factor_data(start_date, end_date, stock, fundamental_factor)
            lastValue, mean, std = get_fundamental_factor_data2(stock, fundamental_factor)
            if lastValue != 0 and mean != 0 and std != 0:
                factorsUsed += 1
            # print(f'for factor: {fundamental_factor} -> lastValue={lastValue}, mean={mean}, std={std}')
            # print((lastValue - mean) / std)
            z += (lastValue - mean) / std
        # TODO: add weighting here -> from Tristan's regressions
        # for econ_factor in econ_factors:
        #     lastValue, mean, std = get_econ_factor_data(econ_factor)
        for econ_factor in econ_factors:
            lastValue, mean, std = get_econ_factor_data(stock, econ_factor)
            if lastValue != 0 and mean != 0 and std != 0:
                factorsUsed += 1
            # print(f'for factor: {econ_factor} -> lastValue={lastValue}, mean={mean}, std={std}')
            # print((lastValue - mean) / std)
            z += (lastValue - mean) / std
        if factorsUsed > 0:
            overall_z_scores.append((z/factorsUsed, stock, sector))
        print(f'z-score for {stock} is {z}')
    overall_z_scores.sort(key=lambda x : x[0])
    n = len(overall_z_scores)
    overall_z_scores = [x for x in overall_z_scores if (x[0] > 3)]
    # top = overall_z_scores[:n//5] # take top and bottom 20%
    # bottom = overall_z_scores[len(overall_z_scores) - n//5 :]
    
    # print(top)
    # print('---------')
    # print(bottom)
    # print(f'n: {n}')
    # print(f'average = total missing / n : {total/n}')
    # print(f'complete: {complete}')
    # print(f'incomplete: {incomplete}')
    return overall_z_scores, n, len(overall_z_scores)


def get_fundamental_factor_data(start_date, end_date, stock, factor):
    stock_factor_data = fundamental_data[fundamental_data['gvkey'] == stock]
    stock_factor_data = stock_factor_data[['gvkey', 'public-date', factor]]
    
    # Convert 'public-date' to datetime
    stock_factor_data['public-date'] = pd.to_datetime(stock_factor_data['public-date'], errors='coerce')
    
    stock_factor_data = stock_factor_data[(stock_factor_data['public-date'] >= start_date) & (stock_factor_data['public-date'] <= end_date)]
    
    # Check if the factor column contains string values
    if stock_factor_data[factor].dtype == 'object':
        stock_factor_data[factor] = stock_factor_data[factor].str.rstrip('%').astype('float')
    
    if stock_factor_data.empty or stock_factor_data[factor].isna().any():
        return 0, 0, 1
    
    mean = stock_factor_data[factor].mean()
    std = stock_factor_data[factor].std()
    lastValue = stock_factor_data[factor].iloc[-1]
    
    return lastValue, mean, std

    
    # get data for the factor for the year
    # filter by stock
    # calculate mean and std for the last year
    # return lastValue, mean, std
'''
def get_fundamental_factor_data2(stock, factor):
    # Convert 'public_date' to datetime format (ensuring consistency)
    
    # fundamental_betas, skipped_fundamental_stocks = get_betas(raw_fundamental_factor_data, 'date')
    
    # Convert factor to string to match column names
    factor = str(factor)
    
    # Filter for the specific stock
    stock_data = fundamental_betas[fundamental_betas['gvkey'] == stock]
    
    # Check if data is available for this stock and factor
    if stock_data.empty or factor not in stock_data.columns or pd.isna(stock_data[factor]).all():
        d[factor] += 1
        return 0, 0, 1
    
    # Calculate statistics for the beta values
    beta_values = stock_data[factor].dropna()
    
    if len(beta_values) == 0:
        d[factor] += 1
        return 0, 0, 1
    
    
    # Get last value (assuming data is chronologically ordered)
    lastValue = stock_data[stock_data['gvkey'] == stock][factor].iloc[-1]
    
    betas_for_factor = fundamental_betas[factor]
    
    mean = float(betas_for_factor.mean(skipna=True))
    std = float(betas_for_factor.std(skipna=True) if len(betas_for_factor) > 1 else 1)
    
    # Return 0, 0, 1 if any of the values are NaN
    if pd.isna(lastValue) or pd.isna(mean) or pd.isna(std):
        return 0, 0, 1
    
    return lastValue, mean, std
'''

def get_fundamental_factor_data2(stock, factor):
    import numpy as np
    # Convert factor to string to match column names
    factor = str(factor)
    
    # Filter for the specific stock
    stock_data = fundamental_betas[fundamental_betas['gvkey'] == stock]
    
    # Check if data is available for this stock and factor
    if stock_data.empty or factor not in stock_data.columns:
        d[factor] += 1
        return 0, 0, 1
    
    try:
        # Get last value for this stock, with direct NaN check
        lastValue = stock_data[factor].iloc[-1]
        if np.isnan(lastValue):
            d[factor] += 1
            return 0, 0, 1
            
        # Get valid values for statistics
        betas_for_factor = fundamental_betas[factor].dropna()
        
        # Calculate mean and std, handling edge cases
        mean = betas_for_factor.mean() 
        std = betas_for_factor.std() if len(betas_for_factor) > 1 else 1
        
        # Final safety check
        if np.isnan(mean) or np.isnan(std) or std == 0:
            return 0, 0, 1
            
        return float(lastValue), float(mean), float(std)
        
    except Exception:
        d[factor] += 1
        return 0, 0, 1


def get_econ_factor_data(stock, factor):
    econ_betas_df = pd.read_csv(r"QEPM\data\econ_beta_results.csv")
    
    # Convert factor to string to match column names
    factor = str(factor)
    
    # Filter for the specific stock
    stock_data = econ_betas_df[econ_betas_df['gvkey'] == stock]
    
    # Check if data is available for this stock and factor
    if stock_data.empty or factor not in stock_data.columns or pd.isna(stock_data[factor]).all():
        d[factor] += 1
        return 0, 0, 1
    
    # Calculate statistics for the beta values
    beta_values = stock_data[factor].dropna()
    
    if len(beta_values) == 0:
        d[factor] += 1
        return 0, 0, 1
    
    # Get last value (assuming data is chronologically ordered)
    lastValue = float(stock_data[stock_data['gvkey'] == stock][factor].iloc[-1])
    
    betas_for_factor = econ_betas_df[factor]
    
    mean = float(betas_for_factor.mean(skipna=True))
    std = float(betas_for_factor.std(skipna=True) if len(betas_for_factor) > 1 else 1)
    
    # Return 0, 0, 1 if any of the values are NaN
    if pd.isna(lastValue) or pd.isna(mean) or pd.isna(std):
        return 0, 0, 1
    
    return lastValue, mean, std
    
# MAIN
start_date = "2010-01-04"
end_date = "2011-12-12"
# for i in range(2011, 2023, 1):
#     end_date = f'{i}-12-12'
res, n, positive_stocks_length = get_z_scores(start_date, end_date, stock_list, factors)
print(f'number of selected stocks {len(res)}')
print(f'number of positive z-score stocks {positive_stocks_length}')
print(f'number of starting stocks: {n}')
    

# Save the results to a CSV file
output_df = pd.DataFrame(res, columns=['z_score', 'stock', 'sector'])
output_df.to_csv(r"QEPM\data\z_scores_all_betas.csv", index=False)
print("Z-scores saved to QEPM\data\z_scores.csv")