from rollingBacktest import backtest
import matplotlib.pyplot as plt

investment = 100

investment_values = []
revival_indices = []

for start_year, end_year in [('2014','2016'), ('2015','2017'), ('2016','2018'), ('2017','2019'), ('2018','2020'), ('2019','2021'), ('2020', '2022'), ('2021', '2023')]:
    print(f"{end_year} Investments:")
    res, isOptimal = backtest('01', start_year, '12', start_year, '01', end_year, '03', end_year)  # 1-year lookback, Q1 invest
    if isOptimal:
        investment = investment * (1 + res)
        if investment <= 0:
            investment = 100
            revival_indices.append(len(investment_values))
    investment_values.append((f"03-{end_year}", investment))
    res, isOptimal = backtest('04', start_year, '03', end_year, '04', end_year, '06', end_year)  # 1-year lookback, Q2 invest
    if isOptimal:
        investment = investment * (1 + res)
        if investment <= 0:
            investment = 100
            revival_indices.append(len(investment_values))
    investment_values.append((f"06-{end_year}", investment))
    res, isOptimal = backtest('07', start_year, '06', end_year, '07', end_year, '09', end_year)  # 1-year lookback, Q3 invest
    if isOptimal:
        investment = investment * (1 + res)
        if investment <= 0:
            investment = 100
            revival_indices.append(len(investment_values))
    investment_values.append((f"09-{end_year}", investment))
    res, isOptimal = backtest('10', start_year, '09', end_year, '10', end_year, '12', end_year)  # 1-year lookback, Q4 invest
    if isOptimal:
        investment = investment * (1 + res)
        if investment <= 0:
            investment = 100
            revival_indices.append(len(investment_values))
    investment_values.append((f"12-{end_year}", investment))
    print(f"Portfolio value = {investment}")
    print(f"Profit since start of 2016 = {investment - 100}")

print(investment_values)

dates, values = zip(*investment_values)

plt.figure(figsize=(12, 6))
plt.plot(dates, values, marker='o')
plt.title('Portfolio Value Over Time')
plt.xlabel('Quarter-End')
plt.ylabel('Portfolio Value')
plt.xticks(rotation=45)
plt.grid(True)

# Add vertical lines at revival points
for idx in revival_indices:
    plt.axvline(x=dates[idx], color='red', linestyle='--', alpha=0.7, label='Revival' if idx == revival_indices[0] else "")

handles, labels = plt.gca().get_legend_handles_labels()
if 'Revival' in labels:
    plt.legend()

plt.tight_layout()
plt.show()