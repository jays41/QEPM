from singleBacktest import backtest
import pandas as pd
import matplotlib.pyplot as plt
from sp500_prices import get_sp500_prices

investment = 100
investment_values = []
revival_indices = []
revival_details = []
previous_weights = None

target_annual_risk = 0.10
LOOKBACK_YEARS = 2

investment_start_year = 2017
investment_end_year = 2017

quarters = [
    ('01', '12', '01', '03'),  # Q1: Use Jan-Dec data, invest Q1
    ('04', '03', '04', '06'),  # Q2: Use Apr-Mar data, invest Q2  
    ('07', '06', '07', '09'),  # Q3: Use Jul-Jun data, invest Q3
    ('10', '09', '10', '12'),  # Q4: Use Oct-Sep data, invest Q4
]

quarter_counter = 0

earliest_possible_start = max(investment_start_year, 2010 + LOOKBACK_YEARS) # data starts from 2010 (as of 12/09/2025)
investment_dates = [str(year) for year in range(earliest_possible_start, investment_end_year + 1)]

for end_year in investment_dates:
    print(f"\n{end_year} Investments:")
    
    for i, (lookback_start_month, lookback_end_month, invest_start_month, invest_end_month) in enumerate(quarters):
        if lookback_start_month == '01' and lookback_end_month == '12':
            lookback_start_year = str(int(end_year) - LOOKBACK_YEARS)
            lookback_end_year = str(int(end_year) - 1)
        else:
            lookback_start_year = str(int(end_year) - 1)
            lookback_end_year = end_year
        
        res, isOptimal, current_weights = backtest(
            target_annual_risk,
            lookback_start_month, lookback_start_year, 
            lookback_end_month, lookback_end_year, 
            invest_start_month, end_year, 
            invest_end_month, end_year,
            previous_weights
        )
        
        if isOptimal:
            investment = investment * (1 + res)
            previous_weights = current_weights
            
            if investment <= 1:  # If investment drops below £1
                print(f"Portfolio revival at quarter {i+1} of {end_year}")
                revival_details.append({
                    'quarter_index': quarter_counter,
                    'date': f"{invest_end_month}-{end_year}",
                    'year': end_year,
                    'quarter': i+1
                })
                investment = 100
                revival_indices.append(len(investment_values))
                previous_weights = None  # Reset weights after revival
        else:
            print(f"Warning: Optimisation failed for Q{i+1} {end_year}")
            
        quarter_end = f"{invest_end_month}-{end_year}"
        investment_values.append((quarter_end, investment))
        quarter_counter += 1
    
    print(f"Portfolio value at end of {end_year}: £{investment:.2f}")
    print(f"Total return since start: {((investment - 100) / 100) * 100:.2f}%")

print(f"\nFinal investment values: {investment_values}")
print(f"Revival points: {revival_indices}")

dates, values = zip(*investment_values)
sp_values = get_sp500_prices(f"{earliest_possible_start}-01-01", f"{end_year}-12-31")

# Convert portfolio dates from MM-YYYY format to datetime objects
portfolio_dates = pd.to_datetime(dates, format='%m-%Y')

# Get S&P 500 values at quarter-end dates that match portfolio dates
sp_aligned = []
for port_date in portfolio_dates:
    # Find the S&P 500 price closest to the portfolio date (within the same month)
    month_mask = (sp_values["Date"].dt.year == port_date.year) & (sp_values["Date"].dt.month == port_date.month)
    month_data = sp_values[month_mask]
    if not month_data.empty:
        # Get the first trading day of that month
        closest_sp_date = month_data["Date"].min()
        closest_sp_price = month_data[month_data["Date"] == closest_sp_date]["Close"].iloc[0]
        sp_aligned.append((closest_sp_date, closest_sp_price))

# Convert S&P 500 to percentage returns starting from 100
if sp_aligned:
    sp_dates, sp_prices = zip(*sp_aligned)
    # Convert string prices to float
    sp_prices_float = [float(price) for price in sp_prices]
    initial_sp_price = sp_prices_float[0]
    # sp_normalized = [100 * (price / initial_sp_price) for price in sp_prices_float]
    sp_normalised = sp_prices_float
    revival_indices_set = set(revival_indices)
    for i in range(len(sp_prices_float)):
        sp_normalised[i] = 100 * (sp_prices_float[i] / initial_sp_price)
        if i in revival_indices_set:
            sp_normalised[i] += 100
    
plt.figure(figsize=(15, 8))
plt.plot(portfolio_dates, values, marker='o', linewidth=2, markersize=4, label='QEPM Portfolio')
if sp_aligned:
    plt.plot(sp_dates, sp_normalised, marker='s', linewidth=2, markersize=3, label='S&P 500')
plt.title(f'Portfolio Value Over Time (QEPM Strategy) | Alpha ={target_annual_risk * 100}%', fontsize=14)
plt.xlabel('Quarter-End', fontsize=12)
plt.ylabel('Portfolio Value (£)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.axhline(y=100, color='gray', linestyle=':', alpha=0.7, label='Initial Investment')

for idx in revival_indices:
    if idx < len(portfolio_dates):
        plt.axvline(x=portfolio_dates[idx], color='red', linestyle='--', alpha=0.7, 
                   label='Revival' if idx == revival_indices[0] else "")

final_return = ((investment - 100) / 100) * 100
plt.text(0.02, 0.98, f'Final Return: {final_return:.2f}%\nRevivals: {len(revival_indices)}', 
         transform=plt.gca().transAxes, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.legend()
plt.tight_layout()
plt.show()

total_capital_injected = 100 + (len(revival_indices) * 100)  # Initial + revivals
print(f"\nPERFORMANCE SUMMARY")
print(f"Initial Investment: £100.00")
print(f"Additional Capital Injected (Revivals): £{len(revival_indices) * 100:.2f}")
print(f"Total Capital Injected: £{total_capital_injected:.2f}")
print(f"Final Portfolio Value: £{investment:.2f}")
print(f"Total Return (ignoring revivals): {((investment - 100) / 100) * 100:.2f}%")
print(f"True Total Return (including revivals): {((investment - total_capital_injected) / total_capital_injected) * 100:.2f}%")
print(f"Number of Revivals: {len(revival_indices)}")

years = len(investment_values) / 4
if years > 0:
    annualised_return_ignoring_revivals = (investment / 100) ** (1/years) - 1
    print(f"Annualised Return (ignoring revivals): {annualised_return_ignoring_revivals * 100:.2f}%")
    
    if total_capital_injected > 0:
        true_annualised_return = (investment / total_capital_injected) ** (1/years) - 1
        print(f"True Annualised Return (including revivals): {true_annualised_return * 100:.2f}%")
        
## need to plot s&p over this and calculate alpha