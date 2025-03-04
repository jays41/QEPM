import pandas as pd
import numpy as np
import os
# import matplotlib.pyplot as plt
from datetime import datetime
from stratified_weights import get_stratified_weights
from preweighting_calculations import get_preweighting_data

def backtest(stock_data, portfolio_df):
    def parse_monthly_price_data(prices_df):
        
        # Drop the gvkey column
        prices_df = prices_df.drop('gvkey', axis=1)
        prices_df = prices_df.drop('return', axis=1)
        
        # Convert date to datetime
        prices_df['date'] = pd.to_datetime(prices_df['date'])
        
        # Extract year and month and create a month-end date
        prices_df['year_month'] = prices_df['date'].dt.to_period('M')
        
        # Group by ticker and year_month, then take the first observation in each group
        monthly_prices = prices_df.groupby(['ticker', 'year_month']).first().reset_index()
        
        # Create a standardized month date (first day of each month) to align all stocks
        monthly_prices['month_date'] = monthly_prices['year_month'].dt.to_timestamp()
        
        # Pivot the dataframe with the standardized month date as index
        prices_pivot = monthly_prices.pivot(index='month_date', columns='ticker', values='close')
        
        # Sort by date for better time series analysis
        prices_pivot = prices_pivot.sort_index()
        
        # print(f"Loaded monthly price data for {len(prices_pivot.columns)} tickers from {prices_pivot.index.min()} to {prices_pivot.index.max()}")
        # print(f"Number of monthly periods: {len(prices_pivot)}")
        
        # Print a sample to verify data structure
        sample_df = prices_pivot.head()
        
        # Count non-NaN values in each row to verify data density
        data_points = sample_df.count(axis=1).tolist()
        # print(f"First few rows have {data_points} data points out of {len(prices_pivot.columns)} possible tickers")
        # print(sample_df.iloc[:, :5])  # Print only first 5 columns for readability
        
        return prices_pivot

    def calculate_returns(data):
        """Calculate monthly returns from price data."""
        # Suppress the warning about fill_method
        with pd.option_context('mode.chained_assignment', None):
            returns = data.pct_change(fill_method=None).dropna()
        return returns

    def parse_portfolio_weights(portfolio_df):
        
        # Check if required columns exist
        if 'ticker' not in portfolio_df.columns or 'weight' not in portfolio_df.columns:
            raise ValueError("CSV file must contain 'ticker' and 'weight' columns")
        
        # Extract tickers and weights
        tickers = portfolio_df['ticker'].tolist()
        weights = portfolio_df['weight'].values
        
        # Calculate long and short exposure
        long_exposure = sum(w for w in weights if w > 0)
        short_exposure = sum(w for w in weights if w < 0)
        net_exposure = long_exposure + short_exposure
        
        # Print portfolio information
        # print(f"Loaded portfolio with {len(tickers)} assets")
        # print(f"Long exposure: {long_exposure:.4f}")
        # print(f"Short exposure: {short_exposure:.4f}")
        # print(f"Net exposure: {net_exposure:.4f}")
        
        return tickers, weights


    def run_backtest(prices, tickers, weights, start_date=None, end_date=None):
        """
        Run a monthly backtest with given prices, tickers and weights.
        Returns are calculated relative to the initial price (buy-in price).
        """
        # Filter prices to include only tickers in our portfolio
        portfolio_tickers = [ticker for ticker in tickers if ticker in prices.columns]
        
        if len(portfolio_tickers) < len(tickers):
            missing = set(tickers) - set(portfolio_tickers)
            # print(f"Warning: {len(missing)} tickers not found in price data: {', '.join(list(missing)[:10])}...")
            
            # Get weights for available tickers
            available_indices = [i for i, ticker in enumerate(tickers) if ticker in portfolio_tickers]
            available_weights = weights[available_indices]
            
            # Calculate exposure for available tickers
            long_sum = sum(w for w in available_weights if w > 0)
            short_sum = sum(w for w in available_weights if w < 0)
            net_sum = long_sum + short_sum
            
            # Check if we have a market-neutral portfolio
            if abs(net_sum) < 0.001:
                # print("Detected market-neutral long-short portfolio (weights sum close to zero)")
                
                # For long-short portfolios, normalize by sum of absolute weights
                abs_sum = np.sum(np.abs(available_weights))
                if abs_sum < 0.001:
                    # print("ERROR: Sum of absolute weights near zero. Cannot normalize properly.")
                    return pd.Series()
                    
                # Scale the weights to maintain long-short balance
                scale_factor = 2 / abs_sum  # Scale so that sum of abs(weights) = 2 (1 long, 1 short)
                available_weights = available_weights * scale_factor
                # print(f"Normalized by scaling with factor {scale_factor:.4f} to maintain long-short balance")
                
                # Verify the new weights
                new_long = sum(w for w in available_weights if w > 0)
                new_short = sum(w for w in available_weights if w < 0)
                # print(f"After normalization: Long {new_long:.4f}, Short {new_short:.4f}, Net {new_long + new_short:.4f}")
                
            else:
                # For non-market neutral, normalize by sum
                if abs(net_sum) < 0.001:
                    # print("ERROR: Available weights sum to near zero and portfolio is not market neutral.")
                    # print("Consider checking your portfolio weights.")
                    return pd.Series()
                    
                available_weights = available_weights / net_sum
                # print(f"Normalized by dividing by net sum: {net_sum:.4f}")
            
            weights = available_weights
            tickers = portfolio_tickers
        
        portfolio_prices = prices[tickers].copy()
        
        # Apply date filters if provided
        if start_date:
            portfolio_prices = portfolio_prices[portfolio_prices.index >= start_date]
        if end_date:
            portfolio_prices = portfolio_prices[portfolio_prices.index <= end_date]
        
        if portfolio_prices.empty:
            # print("ERROR: No price data available after filtering.")
            return pd.Series()
        
        # Store initial prices (buy-in prices)
        initial_prices = portfolio_prices.iloc[0]
        
        # Calculate returns relative to initial prices (absolute returns from buy-in)
        # For longs: (current_price - initial_price) / initial_price
        # For shorts: (initial_price - current_price) / initial_price
        
        # Create a DataFrame to store absolute returns for each stock
        abs_returns = pd.DataFrame(index=portfolio_prices.index, columns=tickers)
        
        # Calculate absolute returns for each ticker based on its initial price
        for ticker in tickers:
            if ticker in portfolio_prices.columns:
                # Get the weight for this ticker
                idx = tickers.index(ticker)
                weight = weights[idx]
                
                # Calculate returns differently for long and short positions
                if weight > 0:  # Long position
                    abs_returns[ticker] = (portfolio_prices[ticker] - initial_prices[ticker]) / initial_prices[ticker]
                else:  # Short position
                    abs_returns[ticker] = (initial_prices[ticker] - portfolio_prices[ticker]) / initial_prices[ticker]
        
        # Apply weights to get portfolio returns
        weighted_returns = pd.DataFrame()
        for ticker in tickers:
            if ticker in abs_returns.columns:
                idx = tickers.index(ticker)
                weight = weights[idx]
                weighted_returns[ticker] = abs_returns[ticker] * weight
        
        # Sum across all tickers to get portfolio returns for each period
        portfolio_absolute_returns = weighted_returns.sum(axis=1)
        
        # The first row will be zero (initial date has no return)
        portfolio_absolute_returns.iloc[0] = 0
        
        if len(portfolio_absolute_returns) == 0:
            # print("ERROR: Portfolio returns calculation resulted in empty series.")
            return pd.Series()
        
        # Calculate performance metrics
        total_return = portfolio_absolute_returns.iloc[-1]
        # Adjust annualization factor for monthly data
        time_period_years = (portfolio_prices.index[-1] - portfolio_prices.index[0]).days / 365.25
        annualized_return = (1 + total_return) ** (1 / time_period_years) - 1 if time_period_years > 0 else 0
        
        # Calculate period-to-period returns for volatility 
        period_returns = portfolio_absolute_returns.diff().dropna()
        volatility = period_returns.std() * np.sqrt(12)  # Monthly to annual
        sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
        
        # Calculate drawdown
        rolling_max = portfolio_absolute_returns.cummax()
        drawdown = (portfolio_absolute_returns - rolling_max) / (1 + rolling_max)
        max_drawdown = drawdown.min()
        
        # Calculate win rate and other statistics
        positive_months = (period_returns > 0).sum()
        win_rate = positive_months / len(period_returns) if len(period_returns) > 0 else 0
        
        # Print results
        # print("\nBacktest Results:")
        # print(f"Period: {portfolio_prices.index.min()} to {portfolio_prices.index.max()}")
        # print(f"Number of periods: {len(portfolio_absolute_returns)}")
        # print(f"Total Absolute Return: {total_return:.4f} ({total_return*100:.2f}%)")
        # print(f"Annualized Return: {annualized_return:.4f} ({annualized_return*100:.2f}%)")
        # print(f"Monthly Volatility: {volatility:.4f} ({volatility*100:.2f}%)")
        # print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        # print(f"Maximum Drawdown: {max_drawdown:.4f} ({max_drawdown*100:.2f}%)")
        # print(f"Win Rate: {win_rate:.4f} ({positive_months}/{len(period_returns)} months)")
        
        return portfolio_absolute_returns


    try:
        ann_return = None
        # Load monthly price data
        prices = parse_monthly_price_data(stock_data)
        
        # Load portfolio weights
        tickers, weights = parse_portfolio_weights(portfolio_df)
        
        # Run backtest
        # print("Running monthly backtest with absolute returns...")
        absolute_returns = run_backtest(prices, tickers, weights)
        
        # Plot results only if we have data
        if not absolute_returns.empty:
            # plt.figure(figsize=(14, 8))
            
            # Plot cumulative returns
            # plt.plot(absolute_returns.index, absolute_returns.values, marker='o', markersize=4, linestyle='-')
            # plt.title('Portfolio Absolute Returns (Relative to Initial Buy-in)', fontsize=14)
            # plt.xlabel('Date', fontsize=12)
            # plt.ylabel('Absolute Return', fontsize=12)
            
            # Format x-axis to show dates more clearly
            # plt.gcf().autofmt_xdate()
            
            # Add grid for better readability
            # plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add horizontal line at y=0
            # plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            
            # Add text with performance metrics
            if len(absolute_returns) > 0:
                total_return = absolute_returns.iloc[-1]
                # Calculate annualized return
                time_period_years = (absolute_returns.index[-1] - absolute_returns.index[0]).days / 365.25
                ann_return = (1 + total_return) ** (1 / time_period_years) - 1 if time_period_years > 0 else 0
                
                # Position text on the plot (adjust as needed)
                # plt.text(
                    # 0.02, 0.95, 
                    # f'Total Return: {total_return*100:.2f}%\nAnn. Return: {ann_return*100:.2f}%', 
                    # # transform=plt.gca().transAxes,
                    # bbox=dict(facecolor='white', alpha=0.7)
                # )
            
            # plt.tight_layout()
            # plt.savefig('absolute_returns_backtest.png')
            # plt.show()
            
        else:
            print("Unable to plot results due to empty return data.")
        
    except Exception as e:
        print(f"Error running backtest: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return ann_return