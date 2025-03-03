from preweighting_calculations import get_preweighting_data
from stratified_weights import get_stratified_weights
from backtest import backtest
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


stock_data, expected_returns, cov_matrix, betas, sectors_array = get_preweighting_data()
results = []
for i in range(1, 40):
    target_annual_risk = i/1000
    portfolio_df, status = get_stratified_weights(stock_data, expected_returns, cov_matrix, betas, sectors_array, target_annual_risk)
    ann_return = backtest(stock_data, portfolio_df)
    print(f'With a target annual risk of {target_annual_risk}, annual return was {ann_return}')
    results.append([target_annual_risk, ann_return, status])


for res in results:
    print(f'{res[0]}: {res[1]*100}, {res[2]}')


plt.figure(figsize=(10, 6))
for res in results:
    risk, ret, status = res
    color = 'green' if status == 'optimal' else 'red'
    plt.scatter(risk, ret*100, color=color, s=50)  # Multiply by 100 to show as percentage

plt.title('Annual Return vs Target Annual Risk')
plt.xlabel('Target Annual Risk')
plt.ylabel('Annual Return (%)')
plt.grid(True, alpha=0.3)


legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Optimal'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Non-optimal')
]
plt.legend(handles=legend_elements, loc='best')


plt.savefig('risk_return_plot.png', dpi=300, bbox_inches='tight')
plt.show()