import pandas as pd

def get_sp500_prices(start_date, end_date):
    data = pd.read_csv(r"QEPM\data\s&p_data.csv")
    data['Date'] = pd.to_datetime(data['Date'])
    filtered_data = data.loc[(data['Date'] >= pd.to_datetime(start_date)) & (data['Date'] <= pd.to_datetime(end_date)), ["Date", "Close"]]
    return filtered_data