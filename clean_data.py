import pandas as pd

def get_rotc(data):
    try:
        rotc = data['opmad'] / (data['debt_assets'] + (1 - row['debt_assets']))
    except:
        print(f'Unable to calculate rotc for {data['ticker']}')
        rotc = None

    return rotc

def get_price_ebitda(data):
    try:
        price_ebitda = data['pe_inc'] * (data['npm'] / data['opmbd'])
    except:
        print(f'Unable to calculate price-ebitda for {data['ticker']}')
        price_ebitda = None
        
    return price_ebitda

def get_cash_conversion_cycle(data):
    try:
        ccc = (365 / data['inv_turn']) + (365 / data['rect_turn']) - (365 / data['pay_turn'])
    except:
        print(f'Unable to calculate cash conversion cycle for {data['ticker']}')
        ccc = None
    
    return ccc

def get_pegy(data):
    try:
        pegy = data['PEG_1yrforward'] / data['divyield']
    except:
        pegy = None
    
    return pegy
    

cleaned_rows = []
df = pd.read_csv('QEPM\data\Data 1500 2010 Start\slmvu0qpmt6cygdi.csv')
cleaned_df = pd.DataFrame(columns=['gvkey', 'ticker', 'dividend-yield', 'EV-EBITDA', 'price-book', 'price-cf', 'price-earnings', 'price-EBITDA', 'price-sales', 'price-earnings-growth', 'price-earnings-growth-dividend-yield', 'cash-ratio', 'current-ratio', 'quick-ratio', 'inventory-turnover', 'receivables-turnover', 'total-asset-turnover', 'cash-conversion-cycle', 'gross-profit-margin', 'net-profit-margin', 'operating-profit-margin', 'return-on-assets', 'return-on-common-equity', 'return-on-total-capital', 'debt-equity', 'total-debt-ratio', 'interest-coverage-ratio'])
for i in range(len(df)):
    row = df.iloc[i]
    new_row = {
        'gvkey': row['gvkey'],
        'ticker': row['TICKER'],
        'dividend-yield': row['divyield'],
        'EV-EBITDA': row['evm'], # enterprice value multiple
        'price-book': row['ptb'],
        'price-cf': row['pcf'],
        'price-earnings': row['pe_inc'], # two options for this, was not sure which one to use: "P/E (Diluted, Excl. EI)": "pe_exi", "P/E (Diluted, Incl. EI)": "pe_inc"
        'price-EBITDA': get_price_ebitda(row),
        'price-sales': row['ps'],
        'price-earnings-growth': row['PEG_1yrforward'], # row['peg_ltg_forward'] or row['peg_1yr_forward'] or row['peg_trailing'], # multiple values available - not sure which we want to use
        'price-earnings-growth-dividend-yield': get_pegy(row) , # can do once we choose which peg ratio to use
        'cash-ratio': row['cash_ratio'],
        'current-ratio': row['curr_ratio'],
        'quick-ratio': row['quick_ratio'],
        'inventory-turnover': row['inv_turn'],
        'receivables-turnover': row['rect_turn'],
        'total-asset-turnover': row['at_turn'], # asset turnover
        'cash-conversion-cycle': get_cash_conversion_cycle(row),
        'gross-profit-margin': row['gpm'],
        'net-profit-margin': row['npm'],
        'operating-profit-margin': row['opmad'] or row['opmbd'], # two options - not sure which to use
        'return-on-assets': row['roa'],
        'return-on-common-equity': row['roe'], # return on equity
        'return-on-total-capital': get_rotc(row),
        'debt-equity': row['de_ratio'],
        'total-debt-ratio': row['debt_at'],
        'interest-coverage-ratio': row['intcov_ratio']
    }
    cleaned_rows.append(new_row)

cleaned_df = pd.DataFrame(cleaned_rows)

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