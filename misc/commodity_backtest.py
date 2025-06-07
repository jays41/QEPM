import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
import os
from tqdm import tqdm  # For the progress bar

# -----------------------------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------------------------
df_stocks = pd.read_csv(
    'data/all_data.csv',
    parse_dates=['date']  # parse 'date' column as datetime
)

df_commodities = pd.read_csv(
    'data/commodities_data.csv',
    parse_dates=['Date'],  # parse 'Date' as datetime
    dayfirst=True          # if your data is DD/MM/YYYY
)

# -----------------------------------------------------------------------------
# 2. RESTRICT BOTH DATASETS TO 2010-01-01 THROUGH 2019-12-31
#    (We do NOT exclude commodities based on coverage.)
# -----------------------------------------------------------------------------
start_date = pd.to_datetime('2010-01-01')
end_date   = pd.to_datetime('2019-12-31')

df_stocks = df_stocks.loc[
    (df_stocks['date'] >= start_date) & 
    (df_stocks['date'] <= end_date)
]
df_commodities = df_commodities.loc[
    (df_commodities['Date'] >= start_date) & 
    (df_commodities['Date'] <= end_date)
]

# Rename columns for easier merging
df_commodities.rename(columns={'Date': 'date', 'Open': 'commodity_price'}, inplace=True)

# -----------------------------------------------------------------------------
# 3. GATHER *ALL* COMMODITIES (NO COVERAGE CHECK)
# -----------------------------------------------------------------------------
commodities_list = df_commodities['Commodity'].unique().tolist()
print(f"Found {len(commodities_list)} commodities (no coverage filter).")

# -----------------------------------------------------------------------------
# 4. LOOP OVER EACH COMMODITY AND EACH YEAR (2010–2019)
#    We do a 'rolling' approach: all data up to that year's end
# -----------------------------------------------------------------------------
years = range(2010, 2020)  # 2010 through 2019
results = []

# For the progress bar, each commodity–year is one "combo"
total_combos = len(commodities_list) * len(years)

with tqdm(total=total_combos, desc="Overall Progress", unit="combos") as pbar:
    for commodity in commodities_list:
        # Subset for this single commodity
        df_commodity_sub = df_commodities[df_commodities['Commodity'] == commodity]

        for year in years:
            # All data up to the end of the current year
            year_end = pd.to_datetime(f"{year}-12-31")

            df_commodity_window = df_commodity_sub[df_commodity_sub['date'] <= year_end]
            df_stocks_window = df_stocks[df_stocks['date'] <= year_end]

            # Merge on date (inner join)
            merged = pd.merge(df_stocks_window, df_commodity_window, on='date', how='inner')
            
            if merged.empty:
                pbar.update(1)
                continue

            # For each stock (gvkey)
            for gvkey_val in merged['gvkey'].unique():
                df_gvkey = merged[merged['gvkey'] == gvkey_val].copy()
                
                # Must have at least 2 data points
                if len(df_gvkey) < 2:
                    continue

                # Attempt OLS: close = alpha + beta * commodity_price
                X = sm.add_constant(df_gvkey['commodity_price'])
                y = df_gvkey['close']
                model = sm.OLS(y, X).fit()

                # Skip if 'const' or 'commodity_price' wasn't estimated
                if 'const' not in model.params or 'commodity_price' not in model.params:
                    continue

                # Spearman's rank correlation
                corr, pval_corr = spearmanr(df_gvkey['close'], df_gvkey['commodity_price'])

                results.append({
                    'year': year,
                    'gvkey': gvkey_val,
                    'commodity': commodity,
                    'alpha': model.params['const'],
                    'beta': model.params['commodity_price'],
                    'p_value_beta': model.pvalues['commodity_price'],
                    'r_squared': model.rsquared,
                    'spearman_corr': corr,
                    'spearman_pval': pval_corr
                })
            
            # Update progress bar after finishing this commodity–year
            pbar.update(1)

# -----------------------------------------------------------------------------
# 5. CONVERT RESULTS TO A DATAFRAME & SAVE
# -----------------------------------------------------------------------------
df_results = pd.DataFrame(results)

# Ensure the "backtest_results" folder exists
os.makedirs('backtest_results', exist_ok=True)

output_path = os.path.join('backtest_results', 'pooled_ols_and_spearman_results.csv')
df_results.to_csv(output_path, index=False)

print(f"\nDone! Results saved to: {output_path}")
