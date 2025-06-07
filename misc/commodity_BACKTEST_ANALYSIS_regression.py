import pandas as pd
import matplotlib.pyplot as plt
import os

# =============================================================================
# 1. LOAD BACKTEST RESULTS
# =============================================================================
df_results = pd.read_csv('backtest_results/pooled_ols_and_spearman_results.csv')
# Expected columns include: year, gvkey, commodity, alpha, beta, r_squared, etc.
print("Loaded results. Columns:\n", df_results.columns)

# =============================================================================
# 2. MERGE IN FIRST-AVAILABLE CLASSIFICATION (gsector, gind, gsubind)
# =============================================================================
df_class = pd.read_csv('data/Company Sector Translation Table.csv')
df_class_sorted = df_class.sort_values(['gvkey', 'datadate'])
df_class_first = df_class_sorted.groupby('gvkey', as_index=False).first()
df_class_first = df_class_first[['gvkey', 'gsector', 'gind', 'gsubind']]

df_merged = pd.merge(df_results, df_class_first, on='gvkey', how='left')
print("After merging classification, columns:\n", df_merged.columns)

# =============================================================================
# 3. TRANSLATE GICS CODES TO ENGLISH NAMES USING THE PROVIDED DICTIONARIES
# =============================================================================

# Provided dictionaries
sector_dict = {
    10: 'Energy', 15: 'Materials', 20: 'Industrials', 25: 'Consumer Discretionary',
    30: 'Consumer Staples', 35: 'Health Care', 40: 'Financials', 45: 'Information Technology',
    50: 'Communication Services', 55: 'Utilities', 60: 'Real Estate'
}

industry_dict = {
    1010: 'Energy', 1510: 'Materials', 2010: 'Capital Goods', 2510: 'Automobiles & Components',
    3010: 'Consumer Staples Distribution & Retail', 3510: 'Health Care Equipment & Services',
    4010: 'Banks', 4510: 'Software & Services', 5010: 'Telecommunication Services',
    5510: 'Utilities', 6010: 'Equity Real Estate Investment Trusts (REITs)'
}

subindustry_dict = {
    10101010: 'Oil & Gas Drilling', 10101020: 'Oil & Gas Equipment & Services', 10102010: 'Integrated Oil & Gas',
    10102020: 'Oil & Gas Exploration & Production', 10102030: 'Oil & Gas Refining & Marketing',
    10102040: 'Oil & Gas Storage & Transportation', 10102050: 'Coal & Consumable Fuels',
    15101010: 'Commodity Chemicals', 15101020: 'Diversified Chemicals', 15101030: 'Fertilizers & Agricultural Chemicals',
    15101040: 'Industrial Gases', 15101050: 'Specialty Chemicals', 15102010: 'Construction Materials',
    15103010: 'Metal, Glass & Plastic Containers', 15103020: 'Paper & Plastic Packaging Products & Materials',
    15104010: 'Aluminum', 15104020: 'Diversified Metals & Mining', 15104025: 'Copper',
    15104030: 'Gold', 15104040: 'Precious Metals & Minerals', 15104045: 'Silver', 15104050: 'Steel',
    15105010: 'Forest Products', 15105020: 'Paper Products', 20101010: 'Aerospace & Defense',
    20102010: 'Building Products', 20103010: 'Construction & Engineering', 20104010: 'Electrical Components & Equipment',
    20104020: 'Heavy Electrical Equipment', 20105010: 'Industrial Conglomerates',
    20106010: 'Construction Machinery & Heavy Transportation Equipment', 20106015: 'Agricultural & Farm Machinery',
    20106020: 'Industrial Machinery & Supplies & Components', 20107010: 'Trading Companies & Distributors',
    20201010: 'Commercial Printing', 20201050: 'Environmental & Facilities Services',
    20201060: 'Office Services & Supplies', 20201070: 'Diversified Support Services', 20201080: 'Security & Alarm Services',
    20202010: 'Human Resource & Employment Services', 20202020: 'Research & Consulting Services',
    20202030: 'Data Processing & Outsourced Services', 20301010: 'Air Freight & Logistics',
    20302010: 'Passenger Airlines', 20303010: 'Marine Transportation', 20304010: 'Rail Transportation',
    20304030: 'Cargo Ground Transportation', 20304040: 'Passenger Ground Transportation',
    20305010: 'Airport Services', 20305020: 'Highways & Railtracks', 20305030: 'Marine Ports & Services',
    25101010: 'Automotive Parts & Equipment', 25101020: 'Tires & Rubber', 25102010: 'Automobile Manufacturers',
    25102020: 'Motorcycle Manufacturers', 25201010: 'Consumer Electronics', 25201020: 'Home Furnishings',
    25201030: 'Homebuilding', 25201040: 'Household Appliances', 25201050: 'Housewares & Specialties',
    25202010: 'Leisure Products', 25203010: 'Apparel, Accessories & Luxury Goods', 25203020: 'Footwear',
    25203030: 'Textiles', 25301010: 'Casinos & Gaming', 25301020: 'Hotels, Resorts & Cruise Lines',
    25301030: 'Leisure Facilities', 25301040: 'Restaurants', 25302010: 'Education Services',
    25302020: 'Specialized Consumer Services', 25501010: 'Distributors', 25503030: 'Broadline Retail',
    25504010: 'Apparel Retail', 25504020: 'Computer & Electronics Retail', 25504030: 'Home Improvement Retail',
    25504040: 'Other Specialty Retail', 25504050: 'Automotive Retail', 25504060: 'Homefurnishing Retail',
    30101010: 'Drug Retail', 30101020: 'Food Distributors', 30101030: 'Food Retail',
    30101040: 'Consumer Staples Merchandise Retail', 30201010: 'Brewers', 30201020: 'Distillers & Vintners',
    30201030: 'Soft Drinks & Non-alcoholic Beverages', 30202010: 'Agricultural Products & Services',
    30202030: 'Packaged Foods & Meats', 30203010: 'Tobacco', 30301010: 'Household Products',
    30302010: 'Personal Care Products', 35101010: 'Health Care Equipment', 35101020: 'Health Care Supplies',
    35102010: 'Health Care Distributors', 35102015: 'Health Care Services', 35102020: 'Health Care Facilities',
    35102030: 'Managed Health Care', 35103010: 'Health Care Technology', 35201010: 'Biotechnology',
    35202010: 'Pharmaceuticals', 35203010: 'Life Sciences Tools & Services', 40101010: 'Diversified Banks',
    40101015: 'Regional Banks', 40201020: 'Diversified Financial Services', 40201030: 'Multi-Sector Holdings',
    40201040: 'Specialized Finance', 40201050: 'Commercial & Residential Mortgage Finance',
    40201060: 'Transaction & Payment Processing Services', 40202010: 'Consumer Finance',
    40203010: 'Asset Management & Custody Banks', 40203020: 'Investment Banking & Brokerage',
    40203030: 'Diversified Capital Markets', 40203040: 'Financial Exchanges & Data',
    40204010: 'Mortgage REITs', 40301010: 'Insurance Broker', 40301020: 'Life & Health Insurance',
    40301030: 'Multi-line Insurance', 40301040: 'Property & Casualty Insurance', 40301050: 'Reinsurance',
    45102010: 'IT Consulting & Other Services', 45102030: 'Internet Services & Infrastructure',
    45103010: 'Application Software', 45103020: 'Systems Software', 45201020: 'Communications Equipment',
    45202030: 'Technology Hardware, Storage & Peripherals', 45203010: 'Electronic Equipment & Instruments',
    45203015: 'Electronic Components', 45203020: 'Electronic Manufacturing Services',
    45203030: 'Technology Distributors', 45301010: 'Semiconductor Materials & Equipment',
    45301020: 'Semiconductors', 50101010: 'Alternative Carriers', 50101020: 'Integrated Telecommunication Services',
    50102010: 'Wireless Telecommunication Services', 50201010: 'Advertising', 50201020: 'Broadcasting',
    50201030: 'Cable & Satellite', 50201040: 'Publishing', 50202010: 'Movies & Entertainment',
    50202020: 'Interactive Home Entertainment', 50203010: 'Interactive Media & Services',
    55101010: 'Electric Utilities', 55102010: 'Gas Utilities', 55103010: 'Multi-Utilities',
    55104010: 'Water Utilities', 55105010: 'Independent Power Producers & Energy Traders',
    55105020: 'Renewable Electricity', 60101010: 'Diversified REITs', 60102510: 'Industrial REITs',
    60103010: 'Hotel & Resort REITs', 60104010: 'Office REITs', 60105010: 'Health Care REITs',
    60106010: 'Multi-Family Residential REITs', 60106020: 'Single-Family Residential REITs',
    60107010: 'Retail REITs', 60108010: 'Other Specialized REITs', 60108020: 'Self-Storage REITs',
    60108030: 'Telecom Tower REITs', 60108040: 'Timber REITs', 60108050: 'Data Center REITs',
    60201010: 'Diversified Real Estate Activities', 60201020: 'Real Estate Operating Companies',
    60201030: 'Real Estate Development', 60201040: 'Real Estate Services'
}

# Convert classification columns to numeric for mapping.
df_merged['gsector'] = pd.to_numeric(df_merged['gsector'], errors='coerce')
df_merged['gind'] = pd.to_numeric(df_merged['gind'], errors='coerce')
df_merged['gsubind'] = pd.to_numeric(df_merged['gsubind'], errors='coerce')

df_merged['gsector_english'] = df_merged['gsector'].map(sector_dict)
df_merged['gind_english'] = df_merged['gind'].map(industry_dict)
df_merged['subind_english'] = df_merged['gsubind'].map(subindustry_dict)

print("\nAfter translating codes:")
print(df_merged[['gsector', 'gsector_english', 'gind', 'gind_english', 'gsubind', 'subind_english']].head())

# =============================================================================
# 4B. SAVE OUTPUT FOLDERS RELATIVE TO "backtest_results"
# =============================================================================
output_dirs = {
    'sector': os.path.join("backtest_results", "regression_by_sector"),
    'industry': os.path.join("backtest_results", "regression_by_industry"),
    'subindustry': os.path.join("backtest_results", "regression_by_subindustry")
}
for key in output_dirs:
    os.makedirs(output_dirs[key], exist_ok=True)

# =============================================================================
# 5. PLOT REGRESSION RESULTS (Average Beta) BY CLASSIFICATION FOR EACH COMMODITY
# =============================================================================
commodities = df_merged['commodity'].unique()

for comm in commodities:
    subset = df_merged[df_merged['commodity'] == comm]
    
    # --- Plot by Sector ---
    group_sector = subset.groupby('gsector_english')['beta'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    plt.bar(group_sector['gsector_english'].astype(str), group_sector['beta'])
    plt.title(f"Average Beta by Sector\nCommodity: {comm}")
    plt.xlabel("Sector")
    plt.ylabel("Average Beta")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs['sector'], f"{comm}_beta_by_sector.png"), dpi=300)
    plt.close()

    # --- Plot by Industry ---
    group_industry = subset.groupby('gind_english')['beta'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    plt.bar(group_industry['gind_english'].astype(str), group_industry['beta'])
    plt.title(f"Average Beta by Industry\nCommodity: {comm}")
    plt.xlabel("Industry")
    plt.ylabel("Average Beta")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs['industry'], f"{comm}_beta_by_industry.png"), dpi=300)
    plt.close()

    # --- Plot by Subindustry ---
    group_subind = subset.groupby('subind_english')['beta'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    plt.bar(group_subind['subind_english'].astype(str), group_subind['beta'])
    plt.title(f"Average Beta by Subindustry\nCommodity: {comm}")
    plt.xlabel("Subindustry")
    plt.ylabel("Average Beta")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs['subindustry'], f"{comm}_beta_by_subindustry.png"), dpi=300)
    plt.close()

print("Regression plots (average Beta) by commodity have been saved under 'backtest_results'.")
