#MASTER INDEX

#
column_descriptions = {
    "InfoCode": "Instrument or internal reference code",
    "dscode": "Security or ticker identifier",
    "MarketDate": "Trading date",
    "close_usd": "Closing price in USD",
    "vwap": "Volume-weighted average price",
    "Volume": "Total trading volume",
    "numshrs": "Number of shares outstanding",
    "mktcap": "Market capitalization"
}

#Financial Ratios
financial_ratios = {
    "Capitalization Ratio": {
        "Variable Name": "capital_ratio",
        "Category": "Capitalization",
        "Formula": (
            "Total Long-term Debt as a fraction of the sum of Total Long-term Debt, "
            "Common/Ordinary Equity and Preferred Stock"
        )
    },
    "Common Equity/Invested Capital": {
        "Variable Name": "equity_invcap",
        "Category": "Capitalization",
        "Formula": "Common Equity as a fraction of Invested Capital"
    },
    "Long-term Debt/Invested Capital": {
        "Variable Name": "debt_invcap",
        "Category": "Capitalization",
        "Formula": "Long-term Debt as a fraction of Invested Capital"
    },
    "Total Debt/Invested Capital": {
        "Variable Name": "totdebt_invcap",
        "Category": "Capitalization",
        "Formula": "Total Debt (Long-term and Current) as a fraction of Invested Capital"
    },
    "Asset Turnover": {
        "Variable Name": "at_turn",
        "Category": "Efficiency",
        "Formula": (
            "Sales as a fraction of the average Total Assets "
            "based on the most recent two periods"
        )
    },
    "Inventory Turnover": {
        "Variable Name": "inv_turn",
        "Category": "Efficiency",
        "Formula": (
            "COGS as a fraction of the average Inventories "
            "based on the most recent two periods"
        )
    },
    "Payables Turnover": {
        "Variable Name": "pay_turn",
        "Category": "Efficiency",
        "Formula": (
            "COGS and change in Inventories as a fraction of the average of "
            "Accounts Payable based on the most recent two periods"
        )
    },
    "Receivables Turnover": {
        "Variable Name": "rect_turn",
        "Category": "Efficiency",
        "Formula": (
            "Sales as a fraction of the average of Accounts Receivables "
            "based on the most recent two periods"
        )
    },
    "Sales/Stockholders Equity": {
        "Variable Name": "sale_equity",
        "Category": "Efficiency",
        "Formula": "Sales per dollar of total Stockholders’ Equity"
    },
    "Sales/Invested Capital": {
        "Variable Name": "sale_invcap",
        "Category": "Efficiency",
        "Formula": "Sales per dollar of Invested Capital"
    },
    "Sales/Working Capital": {
        "Variable Name": "sale_nwc",
        "Category": "Efficiency",
        "Formula": (
            "Sales per dollar of Working Capital, defined as the difference between "
            "Current Assets and Current Liabilities"
        )
    },
    "Inventory/Current Assets": {
        "Variable Name": "invt_act",
        "Category": "Financial Soundness",
        "Formula": "Inventories as a fraction of Current Assets"
    },
    "Receivables/Current Assets": {
        "Variable Name": "rect_act",
        "Category": "Financial Soundness",
        "Formula": "Accounts Receivables as a fraction of Current Assets"
    },
    "Free Cash Flow/Operating Cash Flow": {
        "Variable Name": "fcf_ocf",
        "Category": "Financial Soundness",
        "Formula": (
            "Free Cash Flow as a fraction of Operating Cash Flow, where Free Cash Flow "
            "is Operating Cash Flow minus Capital Expenditures"
        )
    },
    "Operating CF/Current Liabilities": {
        "Variable Name": "ocf_lct",
        "Category": "Financial Soundness",
        "Formula": "Operating Cash Flow as a fraction of Current Liabilities"
    },
    "Cash Flow/Total Debt": {
        "Variable Name": "cash_debt",
        "Category": "Financial Soundness",
        "Formula": "Operating Cash Flow as a fraction of Total Debt"
    },
    "Cash Balance/Total Liabilities": {
        "Variable Name": "cash_lt",
        "Category": "Financial Soundness",
        "Formula": "Cash Balance as a fraction of Total Liabilities"
    },
    "Cash Flow Margin": {
        "Variable Name": "cfm",
        "Category": "Financial Soundness",
        "Formula": (
            "Income before Extraordinary Items and Depreciation "
            "as a fraction of Sales"
        )
    },
    "Short-Term Debt/Total Debt": {
        "Variable Name": "short_debt",
        "Category": "Financial Soundness",
        "Formula": "Short-term Debt as a fraction of Total Debt"
    },
    "Profit Before Depreciation/Current Liabilities": {
        "Variable Name": "profit_lct",
        "Category": "Financial Soundness",
        "Formula": (
            "Operating Income before D&A as a fraction of Current Liabilities"
        )
    },
    "Current Liabilities/Total Liabilities": {
        "Variable Name": "curr_debt",
        "Category": "Financial Soundness",
        "Formula": "Current Liabilities as a fraction of Total Liabilities"
    },
    "Total Debt/EBITDA": {
        "Variable Name": "debt_ebitda",
        "Category": "Financial Soundness",
        "Formula": "Gross Debt as a fraction of EBITDA"
    },
    "Long-term Debt/Book Equity": {
        "Variable Name": "dltt_be",
        "Category": "Financial Soundness",
        "Formula": "Long-term Debt as a fraction of Book Equity"
    },
    "Interest/Average Long-term Debt": {
        "Variable Name": "int_debt",
        "Category": "Financial Soundness",
        "Formula": (
            "Interest as a fraction of the average Long-term Debt "
            "based on the most recent two periods"
        )
    },
    "Interest/Average Total Debt": {
        "Variable Name": "int_totdebt",
        "Category": "Financial Soundness",
        "Formula": (
            "Interest as a fraction of the average Total Debt "
            "based on the most recent two periods"
        )
    },
    "Long-term Debt/Total Liabilities": {
        "Variable Name": "lt_debt",
        "Category": "Financial Soundness",
        "Formula": "Long-term Debt as a fraction of Total Liabilities"
    },
    "Total Liabilities/Total Tangible Assets": {
        "Variable Name": "lt_ppent",
        "Category": "Financial Soundness",
        "Formula": "Total Liabilities as a fraction of Total Tangible Assets"
    },
    "Cash Conversion Cycle (Days)": {
        "Variable Name": "cash_conversion",
        "Category": "Liquidity",
        "Formula": (
            "Inventories per daily COGS plus Account Receivables per daily Sales "
            "minus Account Payables per daily COGS"
        )
    },
    "Cash Ratio": {
        "Variable Name": "cash_ratio",
        "Category": "Liquidity",
        "Formula": "Cash and Short-term Investments as a fraction of Current Liabilities"
    },
    "Current Ratio": {
        "Variable Name": "curr_ratio",
        "Category": "Liquidity",
        "Formula": "Current Assets as a fraction of Current Liabilities"
    },
    "Quick Ratio (Acid Test)": {
        "Variable Name": "quick_ratio",
        "Category": "Liquidity",
        "Formula": (
            "Quick Ratio: (Current Assets - Inventories) "
            "as a fraction of Current Liabilities"
        )
    },
    "Accruals/Average Assets": {
        "Variable Name": "Accrual",
        "Category": "Other",
        "Formula": (
            "Accruals as a fraction of average Total Assets "
            "based on the most recent two periods"
        )
    },
    "Research and Development/Sales": {
        "Variable Name": "RD_SALE",
        "Category": "Other",
        "Formula": "R&D expenses as a fraction of Sales"
    },
    "Avertising Expenses/Sales": {
        "Variable Name": "adv_sale",
        "Category": "Other",
        "Formula": "Advertising Expenses as a fraction of Sales"
    },
    "Labor Expenses/Sales": {
        "Variable Name": "staff_sale",
        "Category": "Other",
        "Formula": "Labor Expenses as a fraction of Sales"
    },
    "Effective Tax Rate": {
        "Variable Name": "efftax",
        "Category": "Profitability",
        "Formula": "Income Tax as a fraction of Pretax Income"
    },
    "Gross Profit/Total Assets": {
        "Variable Name": "GProf",
        "Category": "Profitability",
        "Formula": "Gross Profitability as a fraction of Total Assets"
    },
    "After-tax Return on Average Common Equity": {
        "Variable Name": "aftret_eq",
        "Category": "Profitability",
        "Formula": (
            "Net Income as a fraction of the average of Common Equity "
            "based on the most recent two periods"
        )
    },
    "After-tax Return on Total Stockholders’ Equity": {
        "Variable Name": "aftret_equity",
        "Category": "Profitability",
        "Formula": (
            "Net Income as a fraction of the average of Total Shareholders’ Equity "
            "based on the most recent two periods"
        )
    },
    "After-tax Return on Invested Capital": {
        "Variable Name": "aftret_invcapx",
        "Category": "Profitability",
        "Formula": (
            "Net Income plus Interest Expenses as a fraction of Invested Capital"
        )
    },
    "Gross Profit Margin": {
        "Variable Name": "gpm",
        "Category": "Profitability",
        "Formula": "Gross Profit as a fraction of Sales"
    },
    "Net Profit Margin": {
        "Variable Name": "npm",
        "Category": "Profitability",
        "Formula": "Net Income as a fraction of Sales"
    },
    "Operating Profit Margin After Depreciation": {
        "Variable Name": "opmad",
        "Category": "Profitability",
        "Formula": (
            "Operating Income After Depreciation as a fraction of Sales"
        )
    },
    "Operating Profit Margin Before Depreciation": {
        "Variable Name": "opmbd",
        "Category": "Profitability",
        "Formula": (
            "Operating Income Before Depreciation as a fraction of Sales"
        )
    },
    "Pre-tax Return on Total Earning Assets": {
        "Variable Name": "pretret_earnat",
        "Category": "Profitability",
        "Formula": (
            "Operating Income After Depreciation as a fraction of average Total "
            "Earnings Assets, where TEA = Property, Plant & Equipment + Current Assets"
        )
    },
    "Pre-tax return on Net Operating Assets": {
        "Variable Name": "pretret_noa",
        "Category": "Profitability",
        "Formula": (
            "Operating Income After Depreciation as a fraction of average Net "
            "Operating Assets (NOA), where NOA = Property, Plant & Equipment + "
            "Current Assets - Current Liabilities"
        )
    },
    "Pre-tax Profit Margin": {
        "Variable Name": "ptpm",
        "Category": "Profitability",
        "Formula": "Pretax Income as a fraction of Sales"
    },
    "Return on Assets": {
        "Variable Name": "roa",
        "Category": "Profitability",
        "Formula": (
            "Operating Income Before Depreciation as a fraction of average Total "
            "Assets based on the most recent two periods"
        )
    },
    "Return on Capital Employed": {
        "Variable Name": "roce",
        "Category": "Profitability",
        "Formula": (
            "Earnings Before Interest and Taxes as a fraction of average Capital "
            "Employed, where Capital Employed = Debt (LT + Current) + Common Equity"
        )
    },
    "Return on Equity": {
        "Variable Name": "roe",
        "Category": "Profitability",
        "Formula": (
            "Net Income as a fraction of average Book Equity (sum of Total Parent "
            "Stockholders' Equity + Deferred Taxes + Investment Tax Credit) "
            "based on the most recent two periods"
        )
    },
    "Total Debt/Equity": {
        "Variable Name": "de_ratio",
        "Category": "Solvency",
        "Formula": (
            "Total Liabilities to Shareholders’ Equity (common and preferred)"
        )
    },
    "Total Debt/Total Assets (debt_assets)": {
        "Variable Name": "debt_assets",
        "Category": "Solvency",
        "Formula": "Total Debt as a fraction of Total Assets"
    },
    "Total Debt/Total Assets (debt_at)": {
        "Variable Name": "debt_at",
        "Category": "Solvency",
        "Formula": "Total Liabilities as a fraction of Total Assets"
    },
    "Total Debt/Capital": {
        "Variable Name": "debt_capital",
        "Category": "Solvency",
        "Formula": (
            "Total Debt as a fraction of Total Capital, where Total Debt = Accounts "
            "Payable + Total Debt in Current & Long-term Liabilities, and Total "
            "Capital = Total Debt + Total Equity"
        )
    },
    "After-tax Interest Coverage": {
        "Variable Name": "intcov",
        "Category": "Solvency",
        "Formula": "Multiple of After-tax Income to Interest and Related Expenses"
    },
    "Interest Coverage Ratio": {
        "Variable Name": "intcov_ratio",
        "Category": "Solvency",
        "Formula": "Multiple of EBIT to Interest and Related Expenses"
    },
    "Dividend Payout Ratio": {
        "Variable Name": "dpr",
        "Category": "Valuation",
        "Formula": (
            "Dividends as a fraction of Income Before Extraordinary Items"
        )
    },
    "Forward P/E to 1-year Growth (PEG) ratio": {
        "Variable Name": "PEG_1yrforward",
        "Category": "Valuation",
        "Formula": (
            "Price-to-Earnings, excl. Extraordinary Items (diluted) "
            "to 1-Year EPS Growth rate"
        )
    },
    "Forward P/E to Long-term Growth (PEG) ratio": {
        "Variable Name": "PEG_ltgforward",
        "Category": "Valuation",
        "Formula": (
            "Price-to-Earnings, excl. Extraordinary Items (diluted) "
            "to Long-term EPS Growth rate"
        )
    },
    "Trailing P/E to Growth (PEG) ratio": {
        "Variable Name": "PEG_trailing",
        "Category": "Valuation",
        "Formula": (
            "Price-to-Earnings, excl. Extraordinary Items (diluted) "
            "to 3-Year past EPS Growth"
        )
    },
    "Book/Market": {
        "Variable Name": "bm",
        "Category": "Valuation",
        "Formula": "Book Value of Equity as a fraction of Market Value of Equity"
    },
    "Shillers Cyclically Adjusted P/E Ratio": {
        "Variable Name": "capei",
        "Category": "Valuation",
        "Formula": (
            "Multiple of Market Value of Equity to 5-year moving average of Net Income"
        )
    },
    "Dividend Yield": {
        "Variable Name": "divyield",
        "Category": "Valuation",
        "Formula": "Indicated Dividend Rate as a fraction of Price"
    },
    "Enterprise Value Multiple": {
        "Variable Name": "evm",
        "Category": "Valuation",
        "Formula": "Multiple of Enterprise Value to EBITDA"
    },
    "Price/Cash flow": {
        "Variable Name": "pcf",
        "Category": "Valuation",
        "Formula": (
            "Multiple of Market Value of Equity to Net Cash Flow "
            "from Operating Activities"
        )
    },
    "P/E (Diluted, Excl. EI)": {
        "Variable Name": "pe_exi",
        "Category": "Valuation",
        "Formula": (
            "Price-to-Earnings, excluding Extraordinary Items (diluted)"
        )
    },
    "P/E (Diluted, Incl. EI)": {
        "Variable Name": "pe_inc",
        "Category": "Valuation",
        "Formula": (
            "Price-to-Earnings, including Extraordinary Items (diluted)"
        )
    },
    "Price/Operating Earnings (Basic, Excl. EI)": {
        "Variable Name": "pe_op_basic",
        "Category": "Valuation",
        "Formula": (
            "Price to Operating EPS, excluding Extraordinary Items (Basic)"
        )
    },
    "Price/Operating Earnings (Diluted, Excl. EI)": {
        "Variable Name": "pe_op_dil",
        "Category": "Valuation",
        "Formula": (
            "Price to Operating EPS, excluding Extraordinary Items (Diluted)"
        )
    },
    "Price/Sales": {
        "Variable Name": "ps",
        "Category": "Valuation",
        "Formula": "Multiple of Market Value of Equity to Sales"
    },
    "Price/Book": {
        "Variable Name": "ptb",
        "Category": "Valuation",
        "Formula": (
            "Multiple of Market Value of Equity to Book Value of Equity"
        )
    }
}


#Econ DAta 
econ_codes = {
    592: {
        "DSMnemonic": "USY76...F",
        "IsPadding": 0,
        "IsForecast": 0,
        "IsKeyIndicator": 0,
        "MktDesc": "UNITED STATES",
        "AdjCode": "F",
        "Desc_English": "EXPORT PRICES, ALL COMMODITIES"
    },
    594: {
        "DSMnemonic": "USI76...F",
        "IsPadding": 0,
        "IsForecast": 0,
        "IsKeyIndicator": 0,
        "MktDesc": "UNITED STATES",
        "AdjCode": "F",
        "Desc_English": "EXPORT PRICES, ALL COMMODITIES"
    },
    598: {
        "DSMnemonic": "USQ76.X.F",
        "IsPadding": 0,
        "IsForecast": 0,
        "IsKeyIndicator": 0,
        "MktDesc": "UNITED STATES",
        "AdjCode": "F",
        "Desc_English": "IMPORT PRICES, ALL COMMODITIES"
    },
    918: {
        "DSMnemonic": "USI90C.CB",
        "IsPadding": 0,
        "IsForecast": 0,
        "IsKeyIndicator": 0,
        "MktDesc": "UNITED STATES",
        "AdjCode": "B",
        "Desc_English": "EXPORTS OF GOODS & SERVICES, NOMINAL"
    },
    2121: {
        "DSMnemonic": "USI59MBCB",
        "IsPadding": 0,
        "IsForecast": 0,
        "IsKeyIndicator": 0,
        "MktDesc": "UNITED STATES",
        "AdjCode": "B",
        "Desc_English": "MONETARY AGGREGATES: M2"
    },
    2177: {
        "DSMnemonic": "USY60...",
        "IsPadding": 0,
        "IsForecast": 0,
        "IsKeyIndicator": 0,
        "MktDesc": "UNITED STATES",
        "AdjCode": "R",
        "Desc_English": "INTEREST RATES: CENTRAL BANK POLICY RATE"
    },
    2262: {
        "DSMnemonic": "USI63...F",
        "IsPadding": 0,
        "IsForecast": 0,
        "IsKeyIndicator": 0,
        "MktDesc": "UNITED STATES",
        "AdjCode": "F",
        "Desc_English": "PRODUCER PRICES, ALL COMMODITIES"
    },
    134896: {
        "DSMnemonic": "USCNFCONQ",
        "IsPadding": 0,
        "IsForecast": 0,
        "IsKeyIndicator": 0,
        "MktDesc": "UNITED STATES",
        "AdjCode": "Q",
        "Desc_English": "CONSUMER CONFIDENCE INDEX"
    },
    135149: {
        "DSMnemonic": "USPERINCB",
        "IsPadding": 0,
        "IsForecast": 0,
        "IsKeyIndicator": 0,
        "MktDesc": "UNITED STATES",
        "AdjCode": "B",
        "Desc_English": "PERSONAL INCOME (MONTHLY SERIES) (AR)"
    },
    135153: {
        "DSMnemonic": "USEXPBOPB",
        "IsPadding": 0,
        "IsForecast": 0,
        "IsKeyIndicator": 0,
        "MktDesc": "UNITED STATES",
        "AdjCode": "B",
        "Desc_English": "EXPORTS OF GOODS ON A BALANCE OF PAYMENTS BASIS"
    },
    135155: {
        "DSMnemonic": "USVISBOPB",
        "IsPadding": 0,
        "IsForecast": 0,
        "IsKeyIndicator": 0,
        "MktDesc": "UNITED STATES",
        "AdjCode": "B",
        "Desc_English": "GOODS TRADE BALANCE ON A BALANCE OF PAYMENTS BASIS"
    },
    136169: {
        "DSMnemonic": "USWAGES.D",
        "IsPadding": 0,
        "IsForecast": 0,
        "IsKeyIndicator": 1,
        "MktDesc": "UNITED STATES",
        "AdjCode": "D",
        "Desc_English": "AVG HOURLY REAL EARNINGS - PRIVATE NONFARM INDUSTRIES"
    },
    136966: {
        "DSMnemonic": "USVACTOTP",
        "IsPadding": 0,
        "IsForecast": 0,
        "IsKeyIndicator": 0,
        "MktDesc": "UNITED STATES",
        "AdjCode": "P",
        "Desc_English": "JOB OPENINGS: TOTAL NONFARM (LEVEL)"
    },
    137439: {
        "DSMnemonic": "USCAPUTLQ",
        "IsPadding": 0,
        "IsForecast": 0,
        "IsKeyIndicator": 0,
        "MktDesc": "UNITED STATES",
        "AdjCode": "Q",
        "Desc_English": "CAPACITY UTILIZATION RATE - ALL INDUSTRY"
    },
    148429: {
        "DSMnemonic": "USWAGES.D",
        "IsPadding": 0,
        "IsForecast": 0,
        "IsKeyIndicator": 0,
        "MktDesc": "UNITED STATES",
        "AdjCode": "D",
        "Desc_English": "AHE: ALL EMPS - TOTAL PRIVATE"
    },
    156602: {
        "DSMnemonic": "USCPXFDEE",
        "IsPadding": 0,
        "IsForecast": 0,
        "IsKeyIndicator": 0,
        "MktDesc": "UNITED STATES",
        "AdjCode": "E",
        "Desc_English": "CPI - ALL ITEMS LESS FOOD & ENERGY (CORE)"
    },
    159196: {
        "DSMnemonic": "USY99AACB",
        "IsPadding": 0,
        "IsForecast": 0,
        "IsKeyIndicator": 0,
        "MktDesc": "UNITED STATES",
        "AdjCode": "B",
        "Desc_English": "NATIONAL INCOME, GROSS, NOMINAL"
    },
    200380: {
        "DSMnemonic": "FRFEDFW",
        "IsPadding": 0,
        "IsForecast": 0,
        "IsKeyIndicator": 0,
        "MktDesc": "UNITED STATES",
        "AdjCode": "R",
        "Desc_English": "US FED FUNDS|EFF RATE (W)"
    },
    201525: {
        "DSMnemonic": "USEMPNGME",
        "IsPadding": 0,
        "IsForecast": 0,
        "IsKeyIndicator": 0,
        "MktDesc": "UNITED STATES",
        "AdjCode": "Q",
        "Desc_English": "EMPLOYED - NONFARM INDUSTRIES TOTAL (PAYROLL SURVEY) (CHG)"
    },
    201723: {
        "DSMnemonic": "USHOWN..%",
        "IsPadding": 0,
        "IsForecast": 0,
        "IsKeyIndicator": 0,
        "MktDesc": "UNITED STATES",
        "AdjCode": "R",
        "Desc_English": "HOMEOWNERSHIP RATES - TOTAL US"
    },
    201747: {
        "DSMnemonic": "USHSOETQO",
        "IsPadding": 0,
        "IsForecast": 0,
        "IsKeyIndicator": 0,
        "MktDesc": "UNITED STATES",
        "AdjCode": "O",
        "Desc_English": "EXISTING HOME SALES"
    },
    201782: {
        "DSMnemonic": "USINPERCD",
        "IsPadding": 0,
        "IsForecast": 0,
        "IsKeyIndicator": 0,
        "MktDesc": "UNITED STATES",
        "AdjCode": "D",
        "Desc_English": "DISPOSABLE PERSONAL INCOME PER CAPITA"
    },
    201955: {
        "DSMnemonic": "USM2....A",
        "IsPadding": 0,
        "IsForecast": 0,
        "IsKeyIndicator": 0,
        "MktDesc": "UNITED STATES",
        "AdjCode": "A",
        "Desc_English": "MONEY SUPPLY M2"
    },
    201969: {
        "DSMnemonic": "USM2WSA",
        "IsPadding": 0,
        "IsForecast": 0,
        "IsKeyIndicator": 0,
        "MktDesc": "UNITED STATES",
        "AdjCode": "B",
        "Desc_English": "US MONEY SUPPLY M2|SEASONALLY ADJ."
    },
    202074: {
        "DSMnemonic": "USOPHNF.G",
        "IsPadding": 0,
        "IsForecast": 0,
        "IsKeyIndicator": 0,
        "MktDesc": "UNITED STATES",
        "AdjCode": "G",
        "Desc_English": "OUTPUT PER HOUR - NONFARM BUSINESS SECTOR"
    },
    202150: {
        "DSMnemonic": "USRACBA",
        "IsPadding": 0,
        "IsForecast": 0,
        "IsKeyIndicator": 0,
        "MktDesc": "UNITED STATES",
        "AdjCode": "A",
        "Desc_English": "CENTRAL BANK LIQ| SWAPS (WED)"
    },
    202215: {
        "DSMnemonic": "USRETTOXB",
        "IsPadding": 0,
        "IsForecast": 0,
        "IsKeyIndicator": 0,
        "MktDesc": "UNITED STATES",
        "AdjCode": "B",
        "Desc_English": "RETAIL SALES - TOTAL"
    },
    202600: {
        "DSMnemonic": "USXCPI..E",
        "IsPadding": 0,
        "IsForecast": 0,
        "IsKeyIndicator": 0,
        "MktDesc": "UNITED STATES",
        "AdjCode": "E",
        "Desc_English": "CPI"
    },
    202604: {
        "DSMnemonic": "USXUPNA.Q",
        "IsPadding": 0,
        "IsForecast": 0,
        "IsKeyIndicator": 0,
        "MktDesc": "UNITED STATES",
        "AdjCode": "Q",
        "Desc_English": "UNEMPLOYMENT RATE"
    },
    202605: {
        "DSMnemonic": "USXHST..O",
        "IsPadding": 0,
        "IsForecast": 0,
        "IsKeyIndicator": 0,
        "MktDesc": "UNITED STATES",
        "AdjCode": "O",
        "Desc_English": "HOUSING STARTS"
    },
    202621: {
        "DSMnemonic": "USGDP...C",
        "IsPadding": 0,
        "IsForecast": 0,
        "IsKeyIndicator": 0,
        "MktDesc": "UNITED STATES",
        "AdjCode": "C",
        "Desc_English": "GDP"
    },
    202641: {
        "DSMnemonic": "USCPIYY%R",
        "IsPadding": 0,
        "IsForecast": 0,
        "IsKeyIndicator": 0,
        "MktDesc": "UNITED STATES",
        "AdjCode": "R",
        "Desc_English": "CPI (%YOY)"
    },
    202642: {
        "DSMnemonic": "USPERXTRD",
        "IsPadding": 0,
        "IsForecast": 0,
        "IsKeyIndicator": 0,
        "MktDesc": "UNITED STATES",
        "AdjCode": "D",
        "Desc_English": "PERSONAL INCOME LESS TRANSFER PAYMENTS"
    },
    202661: {
        "DSMnemonic": "USXGDH.D",
        "IsPadding": 0,
        "IsForecast": 0,
        "IsKeyIndicator": 0,
        "MktDesc": "UNITED STATES",
        "AdjCode": "B",
        "Desc_English": "GDP: PER CAPITA (CHG YOY)"
    },
    202664: {
        "DSMnemonic": "USOPICG1F",
        "IsPadding": 0,
        "IsForecast": 0,
        "IsKeyIndicator": 0,
        "MktDesc": "UNITED STATES",
        "AdjCode": "F",
        "Desc_English": "TOTAL PPI CONSUMER GOODS"
    },
    202811: {
        "DSMnemonic": "USPMMM..Q",
        "IsPadding": 0,
        "IsForecast": 0,
        "IsKeyIndicator": 0,
        "MktDesc": "UNITED STATES",
        "AdjCode": "Q",
        "Desc_English": "S&P GLOBAL PMI - MANUFACTURING"
    },
    202813: {
        "DSMnemonic": "USPMIS..R",
        "IsPadding": 0,
        "IsForecast": 0,
        "IsKeyIndicator": 0,
        "MktDesc": "UNITED STATES",
        "AdjCode": "R",
        "Desc_English": "S&P GLOBAL PMI: SERVICES - BUSINESS ACTIVITY"
    }
}
