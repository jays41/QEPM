import pandas as pd
import numpy as np
import cvxpy as cp
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg

# ==========================================
#              DATA SETUP
# ==========================================

# List of technical factors to include
def expected_returns_setup():
    technical_factors_of_interest = ['macd_30']

    # Ensure stock_returns have prct_change
    stock_returns = pd.read_csv(r"QEPM\data\all_data.csv")
    technical_factor_data = pd.read_csv(r"QEPM\data\technical_factors.csv")
    economic_factor_data = pd.read_csv(r"QEPM\data\econ_data.csv")
    fundamental_factor_data = pd.read_csv(r"QEPM\data\stock_fundamental_data.csv")

    # Convert daily stock returns to monthly
    stock_returns['date'] = pd.to_datetime(stock_returns['date'])
    stock_returns['date'] = stock_returns['date'].dt.to_period('M')
    stock_returns['returns'] = stock_returns.groupby('gvkey')['close'].pct_change()
    stock_returns = stock_returns.groupby(['gvkey', 'date'])['returns'].mean().reset_index()

    # ECONOMIC FACTOR DATA MANIPULATION
    economic_factor_data = (
        economic_factor_data.dropna(subset=['Series_Value'])  # Drops rows where Series_Value is NaN
        .groupby(['PeriodDate', 'EcoSeriesID'])['Series_Value']
        .mean()  # You can use sum(), first(), last(), etc.
        .reset_index()
    )

    # Pivot the economic data to wide format
    economic_factor_data.columns = economic_factor_data.columns.str.strip()
    economic_factor_data = economic_factor_data.pivot(index='PeriodDate', columns='EcoSeriesID', values='Series_Value')
    economic_factor_data = economic_factor_data.reset_index()

    # Merge stock returns with economic_factor_data using one-to-many on date
    economic_factor_data['PeriodDate'] = pd.to_datetime(economic_factor_data['PeriodDate']).dt.to_period('M')

    economic_factor_data = (
        economic_factor_data.groupby(['PeriodDate'])
        .mean()  # You can use sum(), first(), last(), etc.
        .reset_index()
    )

    economic_factor_data = stock_returns.merge(economic_factor_data, left_on='date', right_on='PeriodDate', how='left')

    economic_factor_data = (
        economic_factor_data.groupby(['PeriodDate', 'gvkey'])
        .mean()  # You can use sum(), first(), last(), etc.
        .reset_index()
    )
    
    econ_factors = ['gvkey', 'PeriodDate', 200380, 592, 598, 2177, 134896, 202661, 202664, 202811, 202813, 201723, 137439, 148429, 202074, 202600, 202605]
    economic_factor_data = economic_factor_data[econ_factors]

    # Fill missing data for fundamental factors using forward and backward fill
    # fundamental_factor_data = fundamental_factor_data.fillna(method='ffill').fillna(method='bfill')

    # Convert 'public_date' to datetime format (ensuring consistency)
    fundamental_factor_data['public_date'] = pd.to_datetime(fundamental_factor_data['public_date']).dt.to_period('M')

    # Select relevant columns
    columns_to_keep = ['gvkey', 'public_date', 'npm', 'opmad', 'gpm', 'ptpm', 'pretret_earnat', 'equity_invcap', 'debt_invcap', 'capital_ratio', 'invt_act', 'rect_act', 'debt_assets', 'debt_capital', 'cash_ratio', 'adv_sale']
    fundamental_factor_data = fundamental_factor_data[columns_to_keep]

    # Aggregate fundamental data by 'gvkey' and 'public_date'
    fundamental_factor_data = (
        fundamental_factor_data.groupby(['public_date', 'gvkey'])
        .mean()  # You can replace with sum(), first(), etc.
        .reset_index()
    )

    # Merge stock_returns with fundamental factors
    fundamental_factor_data = stock_returns.merge(
        fundamental_factor_data,
        left_on=['date', 'gvkey'],
        right_on=['public_date', 'gvkey'],
        how='left'
    ).drop(columns=['public_date'])  # Drop duplicate column

    # Convert 'date' to Period('M') format
    technical_factor_data['date'] = pd.to_datetime(technical_factor_data['date']).dt.to_period('M')

    # Group by 'gvkey' and 'date' while taking the mean of technical factors
    technical_factor_data = (
        technical_factor_data
        .groupby(['gvkey', 'date'])[technical_factors_of_interest]
        .mean()
        .reset_index()
    )

    # Merge technical factors with fundamental factors correctly
    technical_factor_data = technical_factor_data.merge(
        fundamental_factor_data,  
        left_on=['gvkey', 'date'], 
        right_on=['gvkey', 'date'],  
        how='left'
    )
    
    return stock_returns, economic_factor_data, fundamental_factor_data, technical_factor_data


# ===========================================
#              EXPECTED RETURNS
# ===========================================

"""
 - get_expected_returns(), gets expected returns for economic_factor_data + fundamental_factor_data as df with gvkeys as index
 - get_betas(), returns with gvkeys as column
 - gforecast_values(), returns df with:
        results_df = pd.DataFrame(
            results_list,
            columns=["ID", "ChosenLag_BIC", "ForecastValue", "ForecastDate"]
        )
 
"""

def get_expected_returns(economic_factor_data: pd.DataFrame, 
                         fundamental_factor_data: pd.DataFrame, 
                         technical_factor_data: pd.DataFrame):
    
    technical_factors_of_interest = ['macd_30']

    tau_values, skipped_tau_stocks =  get_taus_momentum(technical_factor_data, technical_factors_of_interest)

    # ======= ECONOMIC FACTORS ======= #
    economic_betas, skipped_economic_stocks = get_betas(economic_factor_data, 'PeriodDate')
    forecasted_economic_factors = forecast_factor(economic_factor_data, 'PeriodDate').set_index("ID")

    # Ensure economic betas are indexed correctly
    economic_betas = economic_betas.set_index('gvkey').drop(columns=['dropped_rows'])

    # Align forecasted values with betas
    economic_betas.columns = economic_betas.columns.astype(str)
    forecasted_economic_factors.index = forecasted_economic_factors.index.astype(str)
    common_factors_economic = economic_betas.columns.intersection(forecasted_economic_factors.index)

    # Extract forecasted values and reshape for matrix multiplication
    forecasted_values_economic = forecasted_economic_factors.loc[common_factors_economic, "ForecastValue"].values.reshape(-1, 1)
    beta_matrix_economic = economic_betas[common_factors_economic].values  # Shape (stocks, factors)

    # Compute expected returns from economic factors
    returns_economic = beta_matrix_economic @ forecasted_values_economic

    # ======= FUNDAMENTAL FACTORS ======= #
    fundamental_betas, skipped_fundamental_stocks = get_betas(fundamental_factor_data, 'date')
    forecasted_fundamental_factors = forecast_factor(fundamental_factor_data, 'date').set_index("ID")

    # Ensure fundamental betas are indexed correctly
    fundamental_betas = fundamental_betas.set_index('gvkey').drop(columns=['dropped_rows'])

    # Align forecasted values with betas
    fundamental_betas.columns = fundamental_betas.columns.astype(str)
    forecasted_fundamental_factors.index = forecasted_fundamental_factors.index.astype(str)
    common_factors_fundamental = fundamental_betas.columns.intersection(forecasted_fundamental_factors.index)

    # Extract forecasted values and reshape
    forecast_values_fundamental = forecasted_fundamental_factors.loc[common_factors_fundamental, "ForecastValue"].values.reshape(-1, 1)
    beta_matrix_fundamental = fundamental_betas[common_factors_fundamental].values  # Shape (stocks, factors)

    # Perform matrix multiplication
    returns_economic = beta_matrix_economic @ forecasted_values_economic
    returns_fundamental = beta_matrix_fundamental @ forecast_values_fundamental

    # Convert back to DataFrame with gvkey index
    returns_economic = pd.DataFrame(returns_economic, index=economic_betas.index, columns=["Expected Return"])
    returns_fundamental = pd.DataFrame(returns_fundamental, index=fundamental_betas.index, columns=["Expected Return"])

    # Reshape to make them contain same stocks
    common_gvkeys = returns_economic.index.intersection(returns_fundamental.index)
    returns_economic = returns_economic.loc[common_gvkeys]
    returns_fundamental = returns_fundamental.loc[common_gvkeys]

    assert returns_economic.shape == returns_fundamental.shape, "Shapes are still mismatched!"

    expected_returns_df = returns_economic + returns_fundamental

    pd.set_option('display.float_format', '{:,.6f}'.format)

    return expected_returns_df, tau_values


def forecast_factor(dataset: pd.DataFrame, date_column_name):
    """
    Forecasts economic factors using AutoRegressive (AR) models, checking BIC and AIC for max 15 Lags.
    
    Parameters:
    - dataset (DataFrame): Contains economic factors in wide format with columns:
        ['PeriodDate', 'Factor1', 'Factor2', ..., 'FactorN']
    
    Returns:
    - A DataFrame with forecasted values for each economic factor.
    """
    
    results_list = []

    # Ensure column names are strings
    dataset.columns = dataset.columns.astype(str)

    # Convert 'PeriodDate' to datetime and ensure it's sorted
    dataset.sort_values(date_column_name, inplace=True)

    # Loop over each economic factor column
    for factor_column in [col for col in dataset.columns if col not in ['returns', 'gvkey', 'month', 'PeriodDate', 'date', 'qdate']]:
        # 1) Subset to this specific factor
        df_factor = dataset[[date_column_name, factor_column]].dropna()
        df_factor.set_index(date_column_name, inplace=True)

        # 2) Determine best lag using BIC (range 1 to max available observations - 1)
        aic_bic_results = []
        max_lag = min(15, len(df_factor) - 1)  # Ensure lag selection is reasonable

        for lag in range(1, max_lag + 1):
            model = AutoReg(df_factor[factor_column], lags=lag, old_names=False).fit()
            aic_bic_results.append((lag, model.aic, model.bic))
        
        # If no valid models, skip this factor
        if not aic_bic_results:
            results_list.append([factor_column, None, None, "Model fit failed"])
            continue
        
        # 3) Select the best model based on BIC
        ab_df = pd.DataFrame(aic_bic_results, columns=["Lag", "AIC", "BIC"])
        best_bic_row = ab_df.loc[ab_df["BIC"].idxmin()]
        best_bic_lag = int(best_bic_row["Lag"])

        # 4) Fit final AutoReg model with chosen lag
        final_model = AutoReg(df_factor[factor_column], lags=best_bic_lag, old_names=False).fit()

        # 5) Forecast one step ahead
        forecast_val = final_model.forecast(steps=1)

        # 6) Forecast date: 1 month after the last available date

        # ARE WE SURE THIS IS FORECASTING EVERY MONTH?
        last_date = df_factor.index[-1]
        forecast_date = last_date.to_timestamp() + pd.DateOffset(months=1)

        # Store results
        results_list.append([
            factor_column,
            best_bic_lag,
            forecast_val.iloc[0],   # Forecasted value
            forecast_date.strftime("%Y-%m-%d")  # Format as string
        ])

    # Convert results to a DataFrame
    results_df = pd.DataFrame(
        results_list,
        columns=["ID", "ChosenLag_BIC", "ForecastValue", "ForecastDate"]
    )

    return results_df

def get_betas(stock_returns, date_name: str):
    """
    Computes betas for economic factors for each stock (gvkey) using Pooled OLS regression.
    
    Parameters:
    - stock_returns (DataFrame): DataFrame with columns ['returns', 'gvkey', 'month', 'PeriodDate', 'Factor1', 'Factor2', ..., 'FactorN']
    
    Returns:
    - DataFrame with betas for each stock (gvkey).
    """
    betas_list = []
    skipped_stocks = []

    
    # Loop over each unique gvkey (stock)
    for gvkey in stock_returns['gvkey'].unique():
        # Subset the data for the current stock (gvkey)
        stock_data = stock_returns[stock_returns['gvkey'] == gvkey].copy()
        
        # Sort stock data by date before aligning X and y
        stock_data = stock_data.sort_values(by=date_name)

        # Select the factors as independent variables (exclude non-factor columns)
        factors = [col for col in stock_data.columns if col not in ['returns', 'gvkey', 'month', 'PeriodDate', 'date', 'public_date']]
        X = stock_data[factors]
        y = stock_data['returns']
        
        # Drop rows with NaN values and align X and y
        initial_count = len(stock_data)
        X = X.dropna()
        y = y.loc[X.index].dropna()
        X = X.loc[y.index]  # Align X with available y values
        dropped_rows = initial_count - len(X)
        

        # Skip if there's not enough data to run the regression
        if len(X) < 2:
            skipped_stocks.append(gvkey)
            continue
        
        # Perform Pooled OLS regression
        model = sm.OLS(y, sm.add_constant(X)).fit()  # Add constant term for the regression
        betas = model.params.values[1:]  # Exclude constant term
        
        # Store the betas for the current stock (gvkey)
        # Store the betas and dropped count for the current stock (gvkey)
        betas_list.append([gvkey, dropped_rows] + list(betas))
        

    # Convert the betas list into a DataFrame
    betas_df = pd.DataFrame(betas_list, columns=['gvkey', 'dropped_rows'] + factors)


    return betas_df, skipped_stocks



def get_taus_momentum(stock_returns, factor_names):
    """
    Computes taus for technical factors using Pooled OLS regression with fundemental factors.
    
    Parameters:
    - stock_returns (DataFrame): DataFrame with columns ['returns', 'gvkey', 'month', 'PeriodDate', 'Factor1', 'Factor2', ..., 'FactorN']
    - factor_names: list of factors that are needed
    
    Returns:
    - DataFrame with betas for each stock (gvkey).
    """
    
    # Take the betas from momentum ONLY and return them
    betas_list = []
    skipped_stocks = []
    
    # Loop over each unique gvkey (stock)
    for gvkey in stock_returns['gvkey'].unique():
        # Subset the data for the current stock (gvkey)
        stock_data = stock_returns[stock_returns['gvkey'] == gvkey].copy()
        
        # Select the factors as independent variables (exclude non-factor columns)
        factors = [col for col in stock_data.columns if col not in ['returns', 'gvkey', 'month', 'PeriodDate', 'date', 'public_date']]
        X = stock_data[factors]
        y = stock_data['returns']
        
        # Drop rows with NaN values and align X and y
        initial_count = len(stock_data)
        X = X.dropna()
        y = y.loc[X.index].dropna()
        X = X.loc[y.index]  # Align X with available y values
        dropped_rows = initial_count - len(X)
                
        # Skip if there's not enough data to run the regression
        # Skip if there's not enough data to run the regression
        if len(X) < 2:
            skipped_stocks.append(gvkey)
            continue
        
        # Perform Pooled OLS regression
        model = sm.OLS(y, sm.add_constant(X)).fit()  # Add constant term for the regression
        betas_of_interest = model.params.values[-len(factor_names):]  # COME BACK HERE _____________________________________________________
        
        # Store the betas for the current stock (gvkey)
        betas_list.append([gvkey, dropped_rows] + list(betas_of_interest))
    
    # Convert the betas list into a DataFrame
    taus_df = pd.DataFrame(betas_list, columns=['gvkey', 'dropped_rows'] + factor_names)

    return taus_df, skipped_stocks

# get_expected_returns(economic_factor_data, fundamental_factor_data, technical_factor_data)
