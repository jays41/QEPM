import pandas as pd

full_dataset = pd.read_csv(r"QEPM\data\stock_fundamental_data.csv")
full_dataset = full_dataset[['gvkey', 'TICKER']]

price_data = pd.read_csv(r"QEPM\data\all_data.csv")

files_to_check = [r"QEPM\results\results_2020_06.csv", r"QEPM\results\results_2020_12.csv", r"QEPM\results\results_2021_06.csv", r"QEPM\results\results_2021_12.csv", r"QEPM\results\results_2022_06.csv", r"QEPM\results\results_2022_12.csv", r"QEPM\results\results_2023_06.csv", r"QEPM\results\results_2023_12.csv"]
dates = ['2020-06-01','2020-12-01','2021-06-01','2021-12-01','2022-06-01','2022-12-01','2023-06-01','2023-12-01',]

res = []

# change this to accomodate the different risk levels
for file, date in zip(files_to_check, dates):
    print(file)
    print(date)
    btr = pd.read_csv(file)
    merged_data = pd.merge(btr, full_dataset, left_on='ticker', right_on='TICKER', how='inner')
    # print(merged_data[['gvkey', 'weight']])
    # Convert 'date' to datetime
    date = pd.to_datetime(date)
    start_date = date - pd.DateOffset(months=6)

    # Filter price_data for relevant dates
    price_data['date'] = pd.to_datetime(price_data['date'])
    filtered_prices = price_data[(price_data['date'] >= start_date) & (price_data['date'] <= date)]

    portfolio_value = 0

    for gvkey, weight in zip(merged_data['gvkey'], merged_data['weight']):
        gvkey_prices = filtered_prices[filtered_prices['gvkey'] == gvkey]

        # Get start price (closest to start_date)
        start_price_row = gvkey_prices[gvkey_prices['date'] >= start_date].sort_values('date').head(1)
        start_price = start_price_row['close'].iloc[0] if not start_price_row.empty else None

        # Get end price (closest to date)
        end_price_row = gvkey_prices[gvkey_prices['date'] >= date].sort_values('date').head(1)
        end_price = end_price_row['close'].iloc[0] if not end_price_row.empty else None

        if start_price is not None and end_price is not None:
            price_diff = end_price - start_price
            portfolio_value += price_diff * weight
    
    print(f'Portfolio value: {portfolio_value}')
    res.append(portfolio_value)

print(res)