import pandas as pd

stock_data = pd.read_csv(r"QEPM\data\stock_prices.csv")
sector_data = pd.read_csv(r"QEPM\data\Company Sectors.csv")
# Select only the 'ticker' and 'sector' columns
stock_data = stock_data[['gvkey', 'date', 'close']]
sector_data = sector_data[['gvkey', 'gind', 'gsector', 'gsubind']]
# Ensure sector_data has unique 'gvkey' and 'gsubind' pairs
sector_data = sector_data.drop_duplicates(subset=['gvkey', 'gsubind'])

# Merge the stock_data and sector_data on 'gvkey' and 'date'/'datadate'
merged_data = pd.merge(stock_data, sector_data, left_on=['gvkey'], right_on=['gvkey'])

# Select the necessary columns
selected_data = merged_data[['gvkey', 'date', 'close', 'gind', 'gsector', 'gsubind']]

# Save the selected data to a CSV file
selected_data.to_csv(r"QEPM\data\all_data.csv", index=False)