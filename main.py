from singleBacktest import backtest
import pandas as pd
import matplotlib.pyplot as plt
from sp500_prices import get_sp500_prices

investment = 100
investment_values = []
revival_indices = []
revival_details = []
previous_weights = None
needs_revival = False  # Flag to track if revival is needed at start of next quarter

target_annual_risk = 0.10
LOOKBACK_YEARS = 2

investment_start_year = 2022
investment_end_year = 2023

quarters = [
    ('01', '12', '01', '03'),  # Q1: Use Jan-Dec data, invest Q1
    ('04', '03', '04', '06'),  # Q2: Use Apr-Mar data, invest Q2  
    ('07', '06', '07', '09'),  # Q3: Use Jul-Jun data, invest Q3
    ('10', '09', '10', '12'),  # Q4: Use Oct-Sep data, invest Q4
]

quarter_counter = 0

earliest_possible_start = max(investment_start_year, 2010 + LOOKBACK_YEARS) # data starts from 2010 (as of 12/09/2025)
investment_dates = [str(year) for year in range(earliest_possible_start, investment_end_year + 1)]

# Add initial investment point at the start date
initial_date = f"01-{earliest_possible_start}"
investment_values.append((initial_date, investment))

for end_year in investment_dates:
    print(f"\n{end_year} Investments:")
    
    for i, (lookback_start_month, lookback_end_month, invest_start_month, invest_end_month) in enumerate(quarters):
        # Check if we need to revive the portfolio at the start of this quarter
        if needs_revival:
            print(f"Portfolio revived to £100 at start of Q{i+1} {end_year}")
            investment = 100
            previous_weights = None
            # Record the revival point in investment_values
            revival_date = f"{invest_start_month}-{end_year}"
            investment_values.append((revival_date, investment))
            revival_indices.append(len(investment_values) - 1)  # Index of the revival we just added
            needs_revival = False
        
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
            
            # Record the quarter-end value (even if it's below £1)
            quarter_end = f"{invest_end_month}-{end_year}"
            investment_values.append((quarter_end, investment))
            
            if investment <= 1:  # If investment drops below £1
                print(f"Portfolio dropped below £1 in Q{i+1} of {end_year} (value: £{investment:.2f})")
                revival_details.append({
                    'quarter_index': quarter_counter,
                    'date': f"{invest_end_month}-{end_year}",
                    'year': end_year,
                    'quarter': i+1
                })
                needs_revival = True  # Flag for revival at start of next quarter
        else:
            print(f"Warning: Optimisation failed for Q{i+1} {end_year}")
            quarter_end = f"{invest_end_month}-{end_year}"
            investment_values.append((quarter_end, investment))
        quarter_counter += 1
    
    print(f"Portfolio value at end of {end_year}: £{investment:.2f}")
    print(f"Total return since start: {((investment - 100) / 100) * 100:.2f}%")

# Handle any pending revival after the last quarter
if needs_revival:
    print("Portfolio would be revived to £100 after the final quarter")

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

# Convert S&P 500 to track same capital injection pattern as portfolio
if sp_aligned:
    sp_dates, sp_prices = zip(*sp_aligned)
    # Convert string prices to float
    sp_prices_float = [float(price) for price in sp_prices]
    
    # Calculate S&P 500 percentage changes
    sp_pct_changes = [0]  # First period has no change
    for i in range(1, len(sp_prices_float)):
        pct_change = (sp_prices_float[i] - sp_prices_float[i-1]) / sp_prices_float[i-1]
        sp_pct_changes.append(pct_change)
    
    # Initialize S&P normalized values starting at 100
    sp_normalised = [100]  # Start with £100 invested
    revival_indices_set = set(revival_indices)
    
    for i in range(1, len(sp_prices_float)):
        if i in revival_indices_set:
            # Revival: add £100 to current S&P value and apply this quarter's return
            sp_normalised.append(sp_normalised[i-1] + 100 + (100 * sp_pct_changes[i]))
        else:
            # Normal period: apply percentage change to previous value
            sp_normalised.append(sp_normalised[i-1] * (1 + sp_pct_changes[i]))
    
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
        
print(f"DEBUG: Investment values length: {len(investment_values)}")
print(f"DEBUG: Revival indices: {revival_indices}")
print(f"DEBUG: Investment values around revivals:")
for idx in revival_indices:
    if 0 <= idx < len(investment_values):
        print(f"  Index {idx}: {investment_values[idx]}")
        
## need to plot s&p over this and calculate alpha