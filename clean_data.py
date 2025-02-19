import pandas as pd

df = pd.read_csv('QEPM\data\Data 1500 2010 Start\slmvu0qpmt6cygdi.csv')
cleaned_df = pd.DataFrame(columns=['gvkey', 'ticker', 'dividend-yield', 'EV-EBITDA', 'price-book', 'price-cf', 'price-earnings', 'price-EBITDA', 'price-sales', 'price-earnings-growth', 'price-earnings-growth-dividend-yield', 'cash-ratio', 'current-ratio', 'quick-ratio', 'inventory-turnover', 'receivables-turnover', 'total-asset-turnover', 'cash-conversion-cycle', 'gross-profit-margin', 'net-profit-margin', 'operating-profit-margin', 'return-on-assets', 'return-on-common-equity', 'return-on-total-capital', 'debt-equity', 'total-debt-ratio', 'interest-coverage-ratio'])
for i in range(len(df)):
    row = df.iloc[i]
    print()
    new_row = pd.DataFrame({
        'gvkey': row['gvkey'],
        'ticker': row['TICKER'],
        'dividend-yield': row['divyield'],
        'EV-EBITDA': ,
        'price-book': row['ptb'],
        'price-cf': row['pcf'],
        'price-earnings': row['pe_inc'], # two options for this, was not sure which one to use: "P/E (Diluted, Excl. EI)": "pe_exi", "P/E (Diluted, Incl. EI)": "pe_inc"
        'price-EBITDA': ,
        'price-sales': row['ps'],
        'price-earnings-growth': ,
        'price-earnings-growth-dividend-yield': ,
        'cash-ratio': row['cash_ratio'],
        'current-ratio': row['curr_ratio'],
        'quick-ratio': row['quick_ratio'],
        'inventory-turnover': row['inv_turn'],
        'receivables-turnover': row['rect_turn'],
        'total-asset-turnover': row['at_turn'], # asset turnover
        'cash-conversion-cycle': ,
        'gross-profit-margin': row['gpm'],
        'net-profit-margin': row['npm'],
        'operating-profit-margin': ,
        'return-on-assets': row['roa'],
        'return-on-common-equity': ,
        'return-on-total-capital': ,
        'debt-equity': ,
        'total-debt-ratio': ,
        'interest-coverage-ratio': row['intcov_ratio']
    })



'''
dividend-yield, EV-EBITDA, price-book, price-cf, price-earnings, price-EBITDA, price-sales, price-earnings-growth, price-earnings-growth-dividend-yield, cash-ratio, current-ratio, quick-ratio, inventory-turnover, receivables-turnover, total-asset-turnover, cash-conversion-cycle, gross-profit-margin, net-profit-margin, operating-profit-margin, return-on-assets, return-on-common-equity, return-on-total-capital, debt-equity, total-debt-ratio, interest-coverage-ratio

Dividend yield (D/P)
Enterprise-value-to- earnings-before- interest-taxes- depreciation-and- amortization (EV/EBITDA)
Price-to-book (P/B)
Price-to-cash-flow (P/CF)
Price-to-earnings (P/E)

Price-to-earnings- before-interest-taxes-depreciation-and-amortization (P/EBITDA)
Price-to-earnings-to- growth (PEG) ratio
Price-to-earnings-to- growth-to-dividend- yield (PEGY) ratio
Research-and-development-to-sales (RNDS)
Price-to-sales (P/S)
Size

Cash-flow-from- operations ratio (CFOR)
Cash ratio (CR)
Current ratio (CUR)
Quick ratio (QR)

Cash-conversion cycle
Cost management index
Cost management index (for banking firms)
Equity turnover (ET)
Fixed-asset turnover (FAT)

Inventory turnover (IT)
Receivables turnover
Total-asset turnover (TAT)

Gross profit margin (GPM)
Net profit margin (NPM)
Operating profit margin (OPM)
Return on net financial assets
Return on net operating assets

Return on assets (ROA)
Return on common equity (ROCE)
Return on owner's equity
Return on total capital (ROTC)

Cash flow coverage ratio (CFCR)
Debt-to-equity (D/E)
Financial leverage ratio
Interest coverage ratio (ICR)

Total debt ratio (TDR)

Trading turnover (TT)
Float capitalization
Number of security owners

'''