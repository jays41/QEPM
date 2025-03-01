import pandas as pd
import numpy as np

# Read the necessary columns from both datasets
daily_prices = pd.read_csv(r"QEPM\data\daily_stock_price_data.csv", 
                         usecols=['InfoCode', 'MarketDate', 'close_usd'])

financials = pd.read_csv(r"QEPM\data\stock_fundamental_data.csv",
                       usecols=['gvkey', 'TICKER'])

print(f"Loaded daily_prices with {len(daily_prices)} rows")
print(f"Loaded financials with {len(financials)} rows")

financials_unique = financials.drop_duplicates(subset=['gvkey']) # remove duplicates so that its just a mapping of gvkey -> ticker
print(f"Financials after removing duplicates: {len(financials_unique)} rows")

# Remove rows with missing close_usd values
daily_prices = daily_prices.dropna(subset=['close_usd'])
print(f"After removing rows with missing close_usd: {len(daily_prices)} rows")

# Convert to strings and pad with zeros so that they match
daily_prices['InfoCode'] = daily_prices['InfoCode'].astype(str).str.strip().str.zfill(6)
financials_unique = financials_unique.copy()  # explicit copy
financials_unique['gvkey'] = financials_unique['gvkey'].astype(str).str.strip().str.zfill(6)

# Rename InfoCode to gvkey to maintain it in the final result
daily_prices.rename(columns={'InfoCode': 'gvkey'}, inplace=True)

# Left join with financials_unique to get TICKER where available
merged_data = daily_prices.merge(financials_unique[['gvkey', 'TICKER']], on='gvkey', how='left')

# Fill missing TICKER values with empty string
merged_data['TICKER'] = merged_data['TICKER'].fillna('')

# Keep only necessary columns
result = merged_data[['gvkey', 'TICKER', 'MarketDate', 'close_usd']]

result['gvkey'] = result['gvkey'].astype(int)
result['MarketDate'] = pd.to_datetime(result['MarketDate'], errors='coerce')
result['close_usd'] = result['close_usd'].round(4)

result.rename(columns={
    'gvkey': 'gvkey',
    'TICKER': 'ticker',
    'MarketDate': 'date',
    'close_usd': 'close'
}, inplace=True)

print(f"Final result has {len(result)} rows")

result.to_csv(r"QEPM\data\stock_prices.csv", index=False)

print("\nProcessing complete! Results saved to stock_prices.csv")