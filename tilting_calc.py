import numpy as np


def tilt_calculate(factor_data):
    # Section 1: x
    # Step 1: Calculate MACD for each stock (dscode)
    # MACD: Difference between fast EMA (span=30) and slow EMA (span=60)
    factor_data['ema_fast_30'] = factor_data.groupby('dscode')['close_usd'].transform(lambda x: x.ewm(span=30, adjust=False).mean())
    factor_data['ema_slow_60'] = factor_data.groupby('dscode')['close_usd'].transform(lambda x: x.ewm(span=60, adjust=False).mean())
    factor_data['macd'] = factor_data['ema_fast_30'] - factor_data['ema_slow_60']

    # Step 2: Calculate the mean and standard deviation of the MACD for each stock
    macd_mean = factor_data['macd'].mean()
    macd_std = factor_data['macd'].std()

    # Avoid dividing by zero if standard deviation = 0
    if macd_std == 0:
        factor_data['z_score'] = 0  # If std is zero, all MACD values are the same, so set Z-score to 0
    else:
        # Step 3: Calculate the Z-score for each stock's MACD value
        factor_data['z_score'] = (factor_data['macd'] - macd_mean) / macd_std

    # Step 4: Min-Max scale the z-scores to the range [0, 100]
    scaled_z_score = (factor_data['z_score'] - factor_data['z_score'].min()) / (factor_data['z_score'].max() - factor_data['z_score'].min()) * 100
    factor_data['scaled_z_score'] = scaled_z_score

    
# Section 2: c
# Step 1: Define piecewise function for c
def piecewise_c(scaled_z_score):
    """
    Define piecewise_c function
    """
    if scaled_z_score > 20:
        return 0  # No tilt for extreme positive momentum
    elif scaled_z_score > 0:
        return 0.05  # c for tilting 'rationally' positive momentum 
    else:
        return 0  # No tilt for negative momentum or zero values

# Step 3: Define the tilt function
def tilt_function(scaled_z_score):
    """
    Tilt function: 1 / (1 + exp(-c * x)) + 0.5
    Uses piecewise_c to define `c` based on the scaled_z_score.
    """
    if scaled_z_score > np.log(pow(3,20)):
        scaled_z_score = 0

    return (1 / (1 + np.exp(-0.05 * scaled_z_score))) + 0.5

# Section 4: Define the constraint function (no tilt for short positions)
def apply_tilt_constraint(factor_data, tau):

    """
    Apply the tilting constraint to the portfolio, only applying tilt to long positions.
    The short positions are unaffected by the tilt.
    """
    
    # Calculate tilt for each stock (apply tilt only to long positions)
    factor_data['tilt'] = factor_data['scaled_z_score'].apply(tilt_function)
    
    # Define the long and short stocks
    long_stocks = factor_data[factor_data['position'] == 1]
    short_stocks = factor_data[factor_data['position'] == -1]
    
    # Calculate the contribution of long positions (tilt applied)
    long_contribution = (long_stocks['weight'] * long_stocks['tau'] * long_stocks['tilt']).sum()

    # Calculate the contribution of short positions (no tilt applied)
    short_contribution = (short_stocks['weight'] * short_stocks['tau']).sum()

    # To ensure the constraint value is zero (dollar-neutral), we need to scale the tilt
    if long_contribution != short_contribution:
        adjustment_factor = short_contribution / long_contribution
        factor_data.loc[factor_data['position'] == 1, 'tilt'] *= adjustment_factor
    
    # Recalculate the long_contribution after adjustment
    long_contribution = (long_stocks['weight'] * long_stocks['tau'] * long_stocks['tilt']).sum()

    # Recalculate the constraint value after adjustment
    constraint_value = long_contribution - short_contribution
    
    return constraint_value



