import pandas as pd

daily_prices = pd.read_csv(r"QEPM\data\daily_stock_price_data.csv", usecols=['InfoCode', 'MarketDate', 'close_usd'])
financials = pd.read_csv(r"QEPM\data\Data 1500 2010 Start\stock_fundamental_data.csv",
                        usecols=['gvkey', 'TICKER'])
daily_prices['InfoCode'] = daily_prices['InfoCode'].astype(str).str.strip().str.zfill(6)
financials['gvkey'] = financials['gvkey'].astype(str).str.strip().str.zfill(6)


print(daily_prices['InfoCode'].dtype)
print(financials['gvkey'].dtype)

# Merge on InfoCode and gvkey
merged_data = daily_prices.merge(financials[['gvkey', 'TICKER']], 
                                 left_on='InfoCode', right_on='gvkey', how='left')

print('missing:')
missing_gvkeys = merged_data[merged_data['gvkey'].isna()]
print(missing_gvkeys[['InfoCode', 'MarketDate']].head(20))  # Change 20 to see more


# Keep only necessary columns
result = merged_data[['gvkey', 'TICKER', 'MarketDate', 'close_usd']]

print('\nresults:')
print(result.head())

result.to_csv(r"QEPM\data\filtered_stock_prices.csv", index=False)