import yfinance as yf
import pandas as pd

def fetch_and_save_data(tickers, start_date, end_date, save_path):
    all_data = []
    for ticker in tickers:
        print(f"Fetching data for: {ticker}")
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
        df.reset_index(inplace=True)
        df['ticker'] = ticker
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.sort_values(by=['ticker', 'Date'], inplace=True)
    combined_df['market_return'] = combined_df.groupby('ticker')['Adj Close'].pct_change()

    
    # Save to CSV
    combined_df.to_csv(save_path, index=False)
    print(f"Data saved to: {save_path}")

if __name__ == "__main__":
    tickers_to_fetch = ["^GSPC"]
    start_date_str = "2010-01-01"
    end_date_str = "2023-12-31"
    output_csv_path = r"QEPM\data\s&p_data.csv"
    
    fetch_and_save_data(
        tickers=tickers_to_fetch,
        start_date=start_date_str,
        end_date=end_date_str,
        save_path=output_csv_path
    )