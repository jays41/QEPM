import pandas as pd

stock_data = pd.read_csv(r"QEPM\data\stock_prices.csv")
# Select only the 'ticker' and 'sector' columns
selected_data = stock_data[['ticker', 'sector']]

# Drop rows where 'sector' is None or empty
selected_data = selected_data[selected_data['sector'].notna() & (selected_data['sector'] != '')]

# Get unique sectors
unique_stocks = selected_data['ticker'].nunique()

# Print the count of the number of unique sectors
print(f"Number of unique sectors: {unique_stocks}")