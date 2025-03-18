import pandas as pd

# Load the CSV file
df = pd.read_csv(r"C:\Users\2same\Quant\QEPM\data\technical_factors.csv")

start_date = "2022-06-20"  # Replace with actual start date
end_date = "2023-06-20"    # Replace with actual end date

# Convert 'date' to datetime format
df['date'] = pd.to_datetime(df['date'])
df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

# List of technical factors
factors = [
    "momentum_15", "sma_15", "rsi_15", "macd_15",
    "momentum_30", "sma_30", "rsi_30", "macd_30",
    "momentum_90", "sma_90", "rsi_90", "macd_90"
]

# Convert factors to numeric (handle missing values)
df[factors] = df[factors].apply(pd.to_numeric, errors='coerce')

# Print summary statistics
"""print("Summary Statistics for Technical Indicators:")
for factor in factors:
    print("____________________________________________________")
    print(f"\n{factor}:")
    print(f"  Mean: {df[factor].mean():.4f}")
    print(f"  Median: {df[factor].median():.4f}")
    print(f"  Min: {df[factor].min():.4f}")
    print(f"  Max: {df[factor].max():.4f}")
    print(f"  Std Dev: {df[factor].std():.4f}")
    print(f"  Missing Values: {df[factor].isna().sum()} \n")"""


print(df["macd_30"].describe())
