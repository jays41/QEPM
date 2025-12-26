import warnings
warnings.filterwarnings('ignore', message='invalid value encountered in reduce')

from collections import defaultdict
import pandas as pd
import numpy as np
from clean_data import get_fundamental_data
from fixed_expected_returns import get_betas
from config import CONFIG

# Load data once
print("Loading data...")
stock_data = pd.read_csv(CONFIG.get_full_path(CONFIG.ALL_DATA_FILE))
fundamental_factor_data = pd.read_csv(CONFIG.get_full_path(CONFIG.FUNDAMENTAL_DATA_FILE))

# Process stock returns correctly
stock_data['date'] = pd.to_datetime(stock_data['date']).dt.to_period('M')
stock_data['returns'] = stock_data.groupby('gvkey')['close'].pct_change()
stock_returns = stock_data.groupby(['gvkey', 'date'])['returns'].mean().reset_index()

# Process fundamental data
fundamental_factor_data['public_date'] = pd.to_datetime(fundamental_factor_data['public_date']).dt.to_period('M')
fundamental_columns = ['gvkey', 'public_date'] + CONFIG.FUNDAMENTAL_FACTORS
fundamental_factor_data = fundamental_factor_data[fundamental_columns]

# Merge and get betas
merged_data = stock_returns.merge(fundamental_factor_data, left_on=['date', 'gvkey'], 
                                 right_on=['public_date', 'gvkey'], how='left').drop(columns=['public_date'])
fundamental_betas, _ = get_betas(merged_data, 'date')

# Prepare selected data for Z-score calculation
selected_data = stock_data[['gvkey', 'date', 'gind']].drop_duplicates().reset_index(drop=True)

missing_data_counter = defaultdict(int)

# Default weights
DEFAULT_FUNDAMENTAL_WEIGHTS = {factor: 1.0 for factor in CONFIG.FUNDAMENTAL_FACTORS}

def get_factor_beta(stock, factor, factor_type='fundamental'):
    """Get beta statistics for fundamental factors"""
    factor = str(factor)
    
    # Use fundamental betas only
    betas_df = fundamental_betas
    factor_prefix = 'fundamental'
    
    if betas_df.empty:
        missing_data_counter[f'{factor_prefix}_{factor}'] += 1
        return 0, 0, 1
    
    stock_data = betas_df[betas_df['gvkey'] == stock]
    
    if stock_data.empty or factor not in betas_df.columns:
        missing_data_counter[f'{factor_prefix}_{factor}'] += 1
        return 0, 0, 1
    
    try:
        last_value = stock_data[factor].iloc[-1]
        if pd.isna(last_value) or np.isinf(last_value):
            missing_data_counter[f'{factor_prefix}_{factor}'] += 1
            return 0, 0, 1
            
        # Calculate cross-sectional statistics
        all_betas = betas_df[factor].replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(all_betas) <= 1:
            missing_data_counter[f'{factor_prefix}_{factor}'] += 1
            return 0, 0, 1
        
        mean = np.nanmean(all_betas)
        std = np.nanstd(all_betas, ddof=1)
        
        if np.isnan(mean) or np.isnan(std) or std == 0 or np.isinf(std):
            missing_data_counter[f'{factor_prefix}_{factor}'] += 1
            return 0, 0, 1
            
        return float(last_value), float(mean), float(std)
        
    except Exception:
        missing_data_counter[f'{factor_prefix}_{factor}'] += 1
        return 0, 0, 1

# Wrapper functions for backward compatibility
def get_fundamental_factor_beta(stock, factor):
    return get_factor_beta(stock, factor, 'fundamental')

def get_economic_factor_beta(stock, factor):
    return get_factor_beta(stock, factor, 'economic')


def calculate_factor_group_z_score(stock, factor_group, factor_weights=None):
    """Calculate Z-score for a group of factors (fundamental only)"""
    if factor_group == 'fundamental':
        factors, get_beta_func, default_weights = CONFIG.FUNDAMENTAL_FACTORS, get_fundamental_factor_beta, DEFAULT_FUNDAMENTAL_WEIGHTS
    else:
        raise ValueError("factor_group must be 'fundamental'")
    
    if factor_weights is None:
        factor_weights = default_weights
    
    # Calculate individual Z-scores for available factors
    valid_factors = {}
    for factor in factors:
        last_value, mean, std = get_beta_func(stock, factor)
        if last_value != 0 and mean != 0 and std != 0:
            z_score = (last_value - mean) / std
            if not (np.isnan(z_score) or np.isinf(z_score)):
                valid_factors[factor] = z_score
    
    if not valid_factors:
        return None, 0, {}, {}
    
    # Normalize weights and calculate weighted Z-score
    available_weights = {factor: factor_weights.get(factor, 1.0) for factor in valid_factors.keys()}
    total_weight = sum(available_weights.values())
    
    if total_weight <= 0:
        return None, 0, {}, {}
    
    normalized_weights = {factor: weight / total_weight for factor, weight in available_weights.items()}
    weighted_z_score = sum(normalized_weights[factor] * z_score for factor, z_score in valid_factors.items())
    
    return weighted_z_score, len(valid_factors), valid_factors, normalized_weights


def calculate_stock_z_score(stock, sector, fundamental_factor_weights=None):
    """Calculate Z-score for a stock using fundamental factors only"""
    # Calculate fundamental factor Z-scores
    fundamental_result = calculate_factor_group_z_score(stock, 'fundamental', fundamental_factor_weights)
    
    if fundamental_result[0] is None:
        return None
    
    z_score, factors_used, individual_z_scores, normalized_weights = fundamental_result
    
    if np.isnan(z_score) or np.isinf(z_score):
        return None
    
    group_details = {
        'fundamental': {
            'z_score': z_score,
            'factors_used': factors_used,
            'individual_z_scores': individual_z_scores,
            'normalised_weights': normalized_weights
        }
    }
    
    return z_score, stock, sector, group_details


def get_z_scores_dataframe(start_date, end_date, z_score_threshold=None, top_percentile=None, 
                          winsorise_percentile=None, z_score_cap=None,
                          fundamental_factor_weights=None):

    if selected_data.empty:
        print("Error: No stock data available for Z-score calculation")
        return pd.DataFrame()

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    stock_list = selected_data.drop_duplicates(subset=['gvkey'])
    
    z_scores = []
    total_stocks = len(stock_list)
    
    print(f"Calculating Z-scores for {total_stocks} stocks...")
    
    for count, (i, row) in enumerate(stock_list.iterrows(), 1):
        stock, sector = row['gvkey'], row['gind']
        
        result = calculate_stock_z_score(stock, sector, fundamental_factor_weights)
        
        if result is not None:
            aggregate_z_score, stock_id, sector_id, group_details = result
            
            fund_z_score = group_details.get('fundamental', {}).get('z_score', np.nan)
            fund_factors_used = group_details.get('fundamental', {}).get('factors_used', 0)
            
            z_scores.append({
                'z_score': aggregate_z_score,
                'stock': stock_id,
                'sector': sector_id,
                'fundamental_z_score': fund_z_score,
                'fundamental_factors_used': fund_factors_used,
                'total_factors_used': fund_factors_used
            })
            
        # if count % 100 == 0:
        #     print(f"Processed {count}/{total_stocks} stocks...")
    
    result_df = pd.DataFrame(z_scores)
    
    if result_df.empty:
        print("Warning: No valid Z-scores calculated!")
        return result_df
    
    result_df = result_df.sort_values('z_score', ascending=False).reset_index(drop=True)
    
    print(f"\nRaw Z-Score Statistics:")
    print(f"Range: {result_df['z_score'].min():.3f} to {result_df['z_score'].max():.3f}")
    print(f"Mean: {result_df['z_score'].mean():.3f}")
    print(f"Std: {result_df['z_score'].std():.3f}")
    print(f"95th percentile: {result_df['z_score'].quantile(0.95):.3f}")
    print(f"5th percentile: {result_df['z_score'].quantile(0.05):.3f}")
    
    print(f"\nFundamental Factor Statistics:")
    if 'fundamental_z_score' in result_df.columns:
        fund_scores = result_df['fundamental_z_score'].dropna()
        if len(fund_scores) > 0:
            print(f"Mean: {fund_scores.mean():.3f}, Std: {fund_scores.std():.3f}")
    
    # Winsorise (as recommended in feedback from Deutsche Bank)
    if winsorise_percentile is not None:
        lower_bound = result_df['z_score'].quantile(winsorise_percentile)
        upper_bound = result_df['z_score'].quantile(1 - winsorise_percentile)
        
        original_count = len(result_df)
        result_df = result_df[
            (result_df['z_score'] >= lower_bound) & 
            (result_df['z_score'] <= upper_bound)
        ].reset_index(drop=True)
        
        print(f"Winsorised at {winsorise_percentile*100}%/{100-winsorise_percentile*100}% - removed {original_count - len(result_df)} extreme outliers")
    
    if z_score_cap is not None:
        original_count = len(result_df)
        result_df = result_df[
            abs(result_df['z_score']) <= z_score_cap
        ].reset_index(drop=True)
        
        print(f"Applied z-score cap of ±{z_score_cap} - removed {original_count - len(result_df)} extreme outliers")
    
    if z_score_threshold is not None:
        result_df = result_df[abs(result_df['z_score']) > z_score_threshold]
        print(f"Filtered to {len(result_df)} stocks with |z_score| > {z_score_threshold}")
    
    if top_percentile is not None:
        n = len(result_df)
        top_n = int(n * top_percentile)
        top_stocks = result_df.head(top_n)
        bottom_stocks = result_df.tail(top_n)
        result_df = pd.concat([top_stocks, bottom_stocks]).reset_index(drop=True)
        print(f"Selected top and bottom {top_percentile*100}% ({top_n} stocks each)")
    
    if not result_df.empty:
        print(f"\nFinal Selection Statistics:")
        print(f"Total stocks selected: {len(result_df)}")
        print(f"Z-score range: {result_df['z_score'].min():.3f} to {result_df['z_score'].max():.3f}")
        print(f"Mean Z-score: {result_df['z_score'].mean():.3f}")
        print(f"Z-score std: {result_df['z_score'].std():.3f}")
        
        positive_scores = len(result_df[result_df['z_score'] > 0])
        negative_scores = len(result_df[result_df['z_score'] < 0])
        print(f"Positive z-scores: {positive_scores}, Negative z-scores: {negative_scores}")
        
        print(f"Average fundamental factors used: {result_df['fundamental_factors_used'].mean():.1f}")
    
    if missing_data_counter:
        print(f"\nMissing Data Summary:")
        for factor, count in sorted(missing_data_counter.items()):
            print(f"{factor}: {count} missing")
    
    return result_df


if __name__ == "__main__":
    start_date = "2015-01-04"
    end_date = "2016-12-12"
    
    print(f"Starting Z-score calculation for period {start_date} to {end_date}")
    print(f"Loaded {len(selected_data)} stocks, {len(fundamental_betas)} fundamental betas")
    
    z_scores_df = get_z_scores_dataframe(
        start_date=start_date,
        end_date=end_date,
        winsorise_percentile=0.05,
        top_percentile=0.2
    )
    
    if z_scores_df.empty:
        print("Error: No Z-scores calculated")
        exit(1)
    
    print(f"Z-score calculation completed: {len(z_scores_df)} stocks selected")
    print(f"Z-score range: {z_scores_df['z_score'].min():.3f} to {z_scores_df['z_score'].max():.3f}")
    
    # Save and display results
    z_scores_df.to_csv("QEPM/data/z_scores_all_betas.csv", index=False)
    print("Results saved to QEPM/data/z_scores_all_betas.csv")
    print("\nTop 10 selected stocks:")
    print(z_scores_df.head(10)[['stock', 'sector', 'z_score', 'total_factors_used']].to_string(index=False))