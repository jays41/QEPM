import pandas as pd
import numpy as np
from tqdm import tqdm  # progress bar

# Read the input data and ensure date is parsed properly
df = pd.read_csv("data/all_data.csv", parse_dates=["date"])

# Sort the data by gvkey and date (important for rolling calculations)
df = df.sort_values(["gvkey", "date"])

# Define the periods for which we compute the indicators
periods = [30]

# Loop over each period with a progress bar to compute the indicators
for p in tqdm(periods, desc="Processing periods"):
    # MACD: Difference between fast EMA (span=p) and slow EMA (span=2*p)
    df[f'ema_fast_{p}'] = df.groupby('gvkey')['close'].transform(lambda x: x.ewm(span=p, adjust=False).mean())
    df[f'ema_slow_{p}'] = df.groupby('gvkey')['close'].transform(lambda x: x.ewm(span=2*p, adjust=False).mean())
    df[f'macd_{p}'] = df[f'ema_fast_{p}'] - df[f'ema_slow_{p}']

# Drop the temporary EMA columns
ema_cols = [col for col in df.columns if col.startswith('ema_fast') or col.startswith('ema_slow')]
df.drop(columns=ema_cols, inplace=True)

# Write the output to "technical_factors.csv"
df.to_csv("technical_factors.csv", index=False)
print("Technical factors have been computed and saved to 'technical_factors.csv'.")
