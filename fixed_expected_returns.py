import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
import warnings
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress specific statsmodels warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", message=".*No supported index is available.*")
warnings.filterwarnings("ignore", message=".*Only PeriodIndexes, DatetimeIndexes.*")

def get_expected_returns(
    economic_factor_data: pd.DataFrame, 
    fundamental_factor_data: pd.DataFrame, 
    technical_factor_data: pd.DataFrame, 
    end_date: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tau_values, skipped_tau_stocks, av_momentum = get_taus_momentum(
        technical_factor_data, end_date
    )
    
    if tau_values.empty:
        logger.warning("No tau values computed - technical factor analysis failed")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    tau_values = tau_values.set_index('gvkey')

    try:
        economic_betas, skipped_economic_stocks = get_betas(
            economic_factor_data, 'PeriodDate'
        )
        forecasted_economic_factors = forecast_factor(
            economic_factor_data, 'PeriodDate'
        ).set_index("ID")
        
        if economic_betas.empty or forecasted_economic_factors.empty:
            logger.warning("Economic factor processing failed")
            returns_economic = pd.DataFrame()
        else:
            economic_betas = economic_betas.set_index('gvkey').drop(
                columns=['dropped_rows'], errors='ignore'
            )
            
            economic_betas.columns = economic_betas.columns.astype(str)
            forecasted_economic_factors.index = forecasted_economic_factors.index.astype(str)
            
            common_factors_economic = economic_betas.columns.intersection(
                forecasted_economic_factors.index
            )
            
            if len(common_factors_economic) == 0:
                logger.warning("No common economic factors found")
                returns_economic = pd.DataFrame()
            else:
                forecasted_values_economic = forecasted_economic_factors.loc[
                    common_factors_economic, "ForecastValue"
                ].values.reshape(-1, 1)
                
                beta_matrix_economic = economic_betas[common_factors_economic].values
                
                if beta_matrix_economic.shape[1] != forecasted_values_economic.shape[0]:
                    logger.error("Matrix dimension mismatch in economic factors")
                    returns_economic = pd.DataFrame()
                else:
                    returns_economic_values = beta_matrix_economic @ forecasted_values_economic
                    returns_economic = pd.DataFrame(
                        returns_economic_values, 
                        index=economic_betas.index, 
                        columns=["Economic_Return"]
                    )
                    
    except Exception as e:
        logger.error(f"Error processing economic factors: {e}")
        returns_economic = pd.DataFrame()

    try:
        fundamental_betas, skipped_fundamental_stocks = get_betas(
            fundamental_factor_data, 'date'
        )
        forecasted_fundamental_factors = forecast_factor(
            fundamental_factor_data, 'date'
        ).set_index("ID")
        
        if fundamental_betas.empty or forecasted_fundamental_factors.empty:
            logger.warning("Fundamental factor processing failed")
            returns_fundamental = pd.DataFrame()
        else:
            fundamental_betas = fundamental_betas.set_index('gvkey').drop(
                columns=['dropped_rows'], errors='ignore'
            )
            
            fundamental_betas.columns = fundamental_betas.columns.astype(str)
            forecasted_fundamental_factors.index = forecasted_fundamental_factors.index.astype(str)
            
            common_factors_fundamental = fundamental_betas.columns.intersection(
                forecasted_fundamental_factors.index
            )
            
            if len(common_factors_fundamental) == 0:
                logger.warning("No common fundamental factors found")
                returns_fundamental = pd.DataFrame()
            else:
                forecasted_values_fundamental = forecasted_fundamental_factors.loc[
                    common_factors_fundamental, "ForecastValue"
                ].values.reshape(-1, 1)
                # Robustness: replace NaN/Inf with 0 before matmul
                forecasted_values_fundamental = np.nan_to_num(
                    forecasted_values_fundamental, nan=0.0, posinf=0.0, neginf=0.0
                )
                
                beta_matrix_fundamental = fundamental_betas[common_factors_fundamental].values
                beta_matrix_fundamental = np.nan_to_num(
                    beta_matrix_fundamental, nan=0.0, posinf=0.0, neginf=0.0
                )
                
                if beta_matrix_fundamental.shape[1] != forecasted_values_fundamental.shape[0]:
                    logger.error("Matrix dimension mismatch in fundamental factors")
                    returns_fundamental = pd.DataFrame()
                else:
                    returns_fundamental_values = beta_matrix_fundamental @ forecasted_values_fundamental
                    returns_fundamental_values = np.nan_to_num(
                        returns_fundamental_values, nan=0.0, posinf=0.0, neginf=0.0
                    )
                    returns_fundamental = pd.DataFrame(
                        returns_fundamental_values, 
                        index=fundamental_betas.index, 
                        columns=["Fundamental_Return"]
                    )
                    
    except Exception as e:
        logger.error(f"Error processing fundamental factors: {e}")
        returns_fundamental = pd.DataFrame()

    common_gvkeys = set(tau_values.index)
    
    if not returns_economic.empty:
        common_gvkeys &= set(returns_economic.index)
    if not returns_fundamental.empty:
        common_gvkeys &= set(returns_fundamental.index)
    
    common_gvkeys = list(common_gvkeys)
    
    if not common_gvkeys:
        logger.error("No common stocks found across all factor types")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    expected_returns_df = pd.DataFrame(
        index=common_gvkeys, 
        columns=["Expected_Return"],
        dtype=float
    )
    expected_returns_df["Expected_Return"] = 0.0
    
    if not returns_economic.empty:
        economic_aligned = returns_economic.loc[common_gvkeys, "Economic_Return"]
        expected_returns_df["Expected_Return"] += economic_aligned
    
    if not returns_fundamental.empty:
        fundamental_aligned = returns_fundamental.loc[common_gvkeys, "Fundamental_Return"]
        expected_returns_df["Expected_Return"] += fundamental_aligned
    
    tau_values_aligned = tau_values.loc[common_gvkeys]
    
    expected_returns_df.rename(columns={"Expected_Return": "Expected Return"}, inplace=True)
    
    logger.info(f"Final dataset contains {len(common_gvkeys)} stocks")
    logger.info(f"Expected returns range: {expected_returns_df['Expected Return'].min():.6f} to {expected_returns_df['Expected Return'].max():.6f}")
    
    return expected_returns_df, tau_values_aligned, av_momentum


def forecast_factor(dataset: pd.DataFrame, date_column_name: str) -> pd.DataFrame:
    results_list = []
    
    dataset.columns = dataset.columns.astype(str)    
    dataset = dataset.copy()
    
    if hasattr(dataset[date_column_name].dtype, 'freq'):
        dataset[date_column_name] = dataset[date_column_name].dt.to_timestamp()
    else:
        dataset[date_column_name] = pd.to_datetime(dataset[date_column_name])
    
    dataset.sort_values(date_column_name, inplace=True)
    
    exclude_columns = {'returns', 'gvkey', 'month', 'PeriodDate', 'date', 'qdate'}
    
    for factor_column in dataset.columns:
        if factor_column in exclude_columns or factor_column == date_column_name:
            continue
            
        try:
            df_factor = dataset[[date_column_name, factor_column]].dropna()
            
            if len(df_factor) < 3:  # Need at least 3 observations
                results_list.append([factor_column, None, 0.0, "Insufficient data"])
                continue
                
            df_factor.set_index(date_column_name, inplace=True)
            
            # Check if factor has variation
            if df_factor[factor_column].std() == 0:
                results_list.append([factor_column, 1, df_factor[factor_column].iloc[-1], "No variation"])
                continue
            
            # Determine best lag using BIC
            max_lag = min(15, len(df_factor) - 2)
            best_model = None
            best_bic = np.inf
            best_lag = 1
            
            for lag in range(1, max_lag + 1):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = AutoReg(df_factor[factor_column], lags=lag, old_names=False).fit()
                        
                        if model.bic < best_bic:
                            best_bic = model.bic
                            best_model = model
                            best_lag = lag
                            
                except Exception as e:
                    logger.debug(f"Failed to fit AR({lag}) for {factor_column}: {e}")
                    continue
            
            if best_model is None:
                # Fall back to simple mean if AR model fails
                mean_value = df_factor[factor_column].mean()
                results_list.append([factor_column, 1, mean_value, "AR model failed, using mean"])
                continue
            
            # Forecast one step ahead
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                forecast_val = best_model.forecast(steps=1)
            
            last_date = df_factor.index[-1]
            forecast_date = last_date + pd.DateOffset(months=1)
            
            results_list.append([
                factor_column,
                best_lag,
                float(forecast_val.iloc[0]) if hasattr(forecast_val, 'iloc') else float(forecast_val),
                forecast_date.strftime("%Y-%m-%d")
            ])
            
        except Exception as e:
            logger.error(f"Error forecasting factor {factor_column}: {e}")
            # Use last available value as fallback
            try:
                last_value = dataset[factor_column].dropna().iloc[-1]
                results_list.append([factor_column, 1, float(last_value), f"Error fallback: {str(e)}"])
            except:
                results_list.append([factor_column, 1, 0.0, f"Complete failure: {str(e)}"])
    
    results_df = pd.DataFrame(
        results_list,
        columns=["ID", "ChosenLag_BIC", "ForecastValue", "ForecastDate"]
    )
    
    return results_df


def get_betas(stock_returns: pd.DataFrame, date_name: str) -> Tuple[pd.DataFrame, List]:
    betas_list = []
    skipped_stocks = []
    
    exclude_columns = {'returns', 'gvkey', 'month', 'PeriodDate', 'date', 'public_date'}
    
    unique_stocks = stock_returns['gvkey'].unique()
    
    for gvkey in unique_stocks:
        try:
            stock_data = stock_returns[stock_returns['gvkey'] == gvkey].copy()
            
            if len(stock_data) < 3:  # Need minimum observations
                skipped_stocks.append(gvkey)
                continue
            
            stock_data = stock_data.sort_values(by=date_name)
            
            factor_columns = [col for col in stock_data.columns if col not in exclude_columns]
            
            if not factor_columns:
                skipped_stocks.append(gvkey)
                continue
            
            X = stock_data[factor_columns]
            y = stock_data['returns']
            
            initial_count = len(stock_data)
            valid_indices = X.dropna().index.intersection(y.dropna().index)
            
            if len(valid_indices) < 2:
                skipped_stocks.append(gvkey)
                continue
            
            X_clean = X.loc[valid_indices]
            y_clean = y.loc[valid_indices]
            dropped_rows = initial_count - len(valid_indices)
            
            # Fit OLS regression
            model = sm.OLS(y_clean, sm.add_constant(X_clean)).fit()
            betas = model.params.values[1:]  # Exclude intercept
            
            betas_list.append([gvkey, dropped_rows] + list(betas))
            
        except Exception as e:
            logger.error(f"Error computing betas for stock {gvkey}: {e}")
            skipped_stocks.append(gvkey)
    
    if not betas_list:
        logger.warning("No valid beta calculations completed")
        return pd.DataFrame(), skipped_stocks
    
    factor_columns = [col for col in stock_returns.columns if col not in exclude_columns]
    betas_df = pd.DataFrame(betas_list, columns=['gvkey', 'dropped_rows'] + factor_columns)
    
    return betas_df, skipped_stocks


def get_taus_momentum(
    stock_returns: pd.DataFrame, 
    end_date: str, 
    factor_names: List[str] = ['macd_30']
) -> Tuple[pd.DataFrame, List, pd.DataFrame]:
    betas_list = []
    skipped_stocks = []
    
    exclude_columns = {'returns', 'gvkey', 'month', 'PeriodDate', 'date', 'public_date'}
    keep_columns = [col for col in stock_returns.columns if col not in exclude_columns]
    keep_columns = [col for col in keep_columns if col not in factor_names] + factor_names
    
    stock_returns_subset = stock_returns[['gvkey', 'date', 'returns'] + keep_columns].copy()
    
    # Ensure end_date comparison uses a Period to match the 'date' Period dtype
    try:
        end_period = pd.Period(end_date, freq='M')
    except Exception:
        # Fallback: attempt to coerce via Timestamp then to_period('M')
        end_period = pd.to_datetime(end_date).to_period('M')
    end_date_data = stock_returns_subset[stock_returns_subset['date'] == end_period]
    factor_averages = end_date_data[factor_names].mean().to_frame().T
    
    for gvkey in stock_returns_subset['gvkey'].unique():
        try:
            stock_data = stock_returns_subset[stock_returns_subset['gvkey'] == gvkey].copy()
            
            if len(stock_data) < 3:
                skipped_stocks.append(gvkey)
                continue
            
            X = stock_data[keep_columns]
            y = stock_data['returns']
            
            valid_indices = X.dropna().index.intersection(y.dropna().index)
            
            if len(valid_indices) < 2:
                skipped_stocks.append(gvkey)
                continue
            
            X_clean = X.loc[valid_indices]
            y_clean = y.loc[valid_indices]
            dropped_rows = len(stock_data) - len(valid_indices)
            
            model = sm.OLS(y_clean, sm.add_constant(X_clean)).fit()
            
            # Extract coefficients for factors of interest
            betas_of_interest = []
            for factor_name in factor_names:
                if factor_name in model.params.index:
                    betas_of_interest.append(model.params[factor_name])
                else:
                    betas_of_interest.append(0.0)
            
            betas_list.append([gvkey, dropped_rows] + betas_of_interest)
            
        except Exception as e:
            logger.error(f"Error computing taus for stock {gvkey}: {e}")
            skipped_stocks.append(gvkey)
    
    if not betas_list:
        logger.warning("No valid tau calculations completed")
        return pd.DataFrame(), skipped_stocks, pd.DataFrame()
    
    taus_df = pd.DataFrame(betas_list, columns=['gvkey', 'dropped_rows'] + factor_names)
    
    return taus_df, skipped_stocks, factor_averages


def get_expected_returns_ending(
    end_date: str, 
    data_path: str = "QEPM/data/",
    returns_freq: str = 'M'
) -> pd.DataFrame:
    try:
        stock_returns = pd.read_csv(f"{data_path}all_data.csv")
        technical_factor_data = pd.read_csv(f"{data_path}technical_factors.csv")
        economic_factor_data = pd.read_csv(f"{data_path}econ_data.csv")
        fundamental_factor_data = pd.read_csv(f"{data_path}stock_fundamental_data.csv")
        
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        return pd.DataFrame()
    
    stock_returns['date'] = pd.to_datetime(stock_returns['date'])
    stock_returns['date'] = stock_returns['date'].dt.to_period('M')
    
    stock_returns['returns'] = stock_returns.groupby('gvkey')['close'].pct_change()
    stock_returns = stock_returns.groupby(['gvkey', 'date'])['returns'].mean().reset_index()
    
    economic_factor_data = economic_factor_data.dropna(subset=['Series_Value'])
    economic_factor_data['PeriodDate'] = pd.to_datetime(economic_factor_data['PeriodDate'])
    economic_factor_data['Month'] = economic_factor_data['PeriodDate'].dt.to_period('M')
    
    economic_factor_data = (
        economic_factor_data
        .groupby(['Month', 'EcoSeriesID'])['Series_Value']
        .mean()
        .reset_index()
        .rename(columns={'Month': 'PeriodDate'})
    )
    
    economic_factor_data = economic_factor_data.pivot(
        index='PeriodDate', 
        columns='EcoSeriesID', 
        values='Series_Value'
    ).reset_index()
    
    economic_factor_data = stock_returns.merge(
        economic_factor_data, 
        left_on='date', 
        right_on='PeriodDate', 
        how='left'
    )
    
    fundamental_factor_data['public_date'] = pd.to_datetime(
        fundamental_factor_data['public_date']
    ).dt.to_period('M')
    
    fundamental_columns = [
        'gvkey', 'public_date', 'npm', 'opmad', 'gpm', 'ptpm', 'pretret_earnat',
        'equity_invcap', 'debt_invcap', 'capital_ratio', 'invt_act', 'rect_act',
        'debt_assets', 'debt_capital', 'cash_ratio', 'adv_sale'
    ]
    
    fundamental_factor_data = fundamental_factor_data[fundamental_columns]
    
    fundamental_factor_data = stock_returns.merge(
        fundamental_factor_data,
        left_on=['date', 'gvkey'],
        right_on=['public_date', 'gvkey'],
        how='left'
    ).drop(columns=['public_date'])
    
    technical_factor_data['date'] = pd.to_datetime(technical_factor_data['date']).dt.to_period('M')
    
    technical_factor_data = (
        technical_factor_data
        .groupby(['gvkey', 'date'])[['macd_30']]
        .mean()
        .reset_index()
    )
    
    technical_factor_data = technical_factor_data.merge(
        fundamental_factor_data,
        on=['gvkey', 'date'],
        how='left'
    )
    
    # Convert filter end date to monthly Period for consistent comparisons
    try:
        end_period = pd.Period(end_date, freq='M')
    except Exception:
        end_period = pd.to_datetime(end_date).to_period('M')

    economic_factor_data = economic_factor_data[economic_factor_data['date'] <= end_period]
    fundamental_factor_data = fundamental_factor_data[fundamental_factor_data['date'] <= end_period]
    technical_factor_data = technical_factor_data[technical_factor_data['date'] <= end_period]
    
    expected_returns_df, tau_values, av_momentum = get_expected_returns(
        economic_factor_data, 
        fundamental_factor_data, 
        technical_factor_data, 
        end_date
    )
    
    if expected_returns_df.empty:
        logger.error("Expected returns calculation failed")
        return pd.DataFrame()
    
    # expected_returns_df["Expected Return"] is monthly by construction
    # If quarterly requested, compound 3 months into one quarter
    if returns_freq == 'Q':
        er_m = expected_returns_df["Expected Return"].astype(float)
        expected_returns_df["Expected Return"] = (1.0 + er_m) ** 3 - 1.0
        cap = 0.35  # quarterly hard cap
        cap_label = 'quarterly'
    else:
        cap = 0.15  # monthly hard cap
        cap_label = 'monthly'

    # Winsorize expected returns to reduce the effect of outliers
    er_series = expected_returns_df["Expected Return"].astype(float)
    # Soft bounds from percentiles
    q_low = er_series.quantile(0.01)
    q_high = er_series.quantile(0.99)
    # Time-scale specific hard cap
    hard_cap = cap
    low_cap = max(float(q_low), -hard_cap)
    high_cap = min(float(q_high), hard_cap)
    expected_returns_df["Expected Return"] = er_series.clip(lower=low_cap, upper=high_cap)
    logger.info(f"Winsorized expected returns to [{low_cap:.4%}, {high_cap:.4%}] {cap_label}")
    
    # Remove extreme values
    expected_returns_df = expected_returns_df.replace([np.inf, -np.inf], np.nan)
    expected_returns_df = expected_returns_df.dropna(subset=["Expected Return"])
    expected_returns_df = expected_returns_df[
        expected_returns_df["Expected Return"].abs() < 0.1
    ]
    
    expected_returns_df = expected_returns_df.reset_index()
    expected_returns_df.rename(columns={'index': 'gvkey'}, inplace=True)
    expected_returns_df = expected_returns_df.reset_index(drop=True)
    
    logger.info(
        f"Final expected returns computed for {len(expected_returns_df)} stocks; "
        f"range: {expected_returns_df['Expected Return'].min():.6f} to {expected_returns_df['Expected Return'].max():.6f}"
    )
    
    return expected_returns_df