import pandas as pd
import numpy as np

# Date Parsing

df['MarketDate'] = pd.to_datetime(df['MarketDate'], format='%d/%m/%Y')
df.sort_values(by=['dscode', 'MarketDate'], inplace=True)
df.reset_index(drop=True, inplace=True)

#BollingerBand Calc

window_bb = 20
num_std = 2
df['BB_Middle'] = df.groupby('dscode')['close_usd'].transform(lambda x: x.rolling(window_bb).mean())
df['BB_Std'] = df.groupby('dscode')['close_usd'].transform(lambda x: x.rolling(window_bb).std())
df['BB_Upper'] = df['BB_Middle'] + num_std * df['BB_Std']
df['BB_Lower'] = df['BB_Middle'] - num_std * df['BB_Std']

#Channel Breakout Bands

window_cb = 20
df['channel_high'] = df.groupby('dscode')['close_usd'].transform(lambda x: x.rolling(window_cb).max())
df['channel_low'] = df.groupby('dscode')['close_usd'].transform(lambda x: x.rolling(window_cb).min())
df['channel_breakout_up'] = df['close_usd'] > df['channel_high'].shift(1)
df['channel_breakout_down'] = df['close_usd'] < df['channel_low'].shift(1)

#Lowprice Indic

low_price_threshold = 5.0
df['LowPriceFlag'] = (df['close_usd'] < low_price_threshold).astype(int)

#Momentum

momentum_period = 10
df['Momentum_10d'] = df.groupby('dscode')['close_usd'].transform(lambda x: x.pct_change(momentum_period))

#Moving avg

ma_short_window = 50
ma_long_window = 200
df['SMA_50'] = df.groupby('dscode')['close_usd'].transform(lambda x: x.rolling(ma_short_window).mean())
df['SMA_200'] = df.groupby('dscode')['close_usd'].transform(lambda x: x.rolling(ma_long_window).mean())

#MACD indic

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

df['EMA_12'] = df.groupby('dscode')['close_usd'].transform(lambda x: ema(x, span=12))
df['EMA_26'] = df.groupby('dscode')['close_usd'].transform(lambda x: ema(x, span=26))
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['MACD_Signal'] = df.groupby('dscode')['MACD'].transform(lambda x: ema(x, span=9))
df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

#RSI Calc

def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

df['RSI_14'] = df.groupby('dscode')['close_usd'].transform(lambda x: compute_rsi(x, 14))

#Support Resis

support_res_window = 30
df['Support_30'] = df.groupby('dscode')['close_usd'].transform(lambda x: x.rolling(support_res_window).min())
df['Resistance_30'] = df.groupby('dscode')['close_usd'].transform(lambda x: x.rolling(support_res_window).max())

if 'odd_lot_sales' in df.columns and 'odd_lot_purchases' in df.columns:
    df['OLBI'] = df['odd_lot_sales'] / df['odd_lot_purchases']
