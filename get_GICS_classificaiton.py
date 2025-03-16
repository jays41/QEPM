import pandas as pd

# ---------------------------
# Table 1: Energy, Materials, Industrials
# ---------------------------
# Energy classification
energy_data = [
    (10, "Energy", 1010, "Energy"),
    (101010, "Energy Equipment & Services", 10101010, "Oil & Gas Drilling"),
    (101010, "Energy Equipment & Services", 10101020, "Oil & Gas Equipment & Services"),
    (101020, "Oil, Gas & Consumable Fuels", 10102010, "Integrated Oil & Gas"),
    (101020, "Oil, Gas & Consumable Fuels", 10102020, "Oil & Gas Exploration & Production"),
    (101020, "Oil, Gas & Consumable Fuels", 10102030, "Oil & Gas Refining & Marketing"),
    (101020, "Oil, Gas & Consumable Fuels", 10102040, "Oil & Gas Storage & Transportation"),
    (101020, "Oil, Gas & Consumable Fuels", 10102050, "Coal & Consumable Fuels"),
]

# Materials classification
materials_data = [
    (15, "Materials", 1510, "Materials"),
    (151010, "Chemicals", 15101010, "Commodity Chemicals"),
    (151010, "Chemicals", 15101020, "Diversified Chemicals"),
    (151010, "Chemicals", 15101030, "Fertilizers & Agricultural Chemicals"),
    (151010, "Chemicals", 15101040, "Industrial Gases"),
    (151010, "Chemicals", 15101050, "Specialty Chemicals"),
    (151020, "Construction Materials", 15102010, "Construction Materials"),
    (151030, "Containers & Packaging", 15103010, "Metal, Glass & Plastic Containers"),
    (151030, "Containers & Packaging", 15103020, "Paper & Plastic Packaging Products & Materials"),
    (151040, "Metals & Mining", 15104010, "Aluminum"),
    (151040, "Metals & Mining", 15104020, "Diversified Metals & Mining"),
    (151040, "Metals & Mining", 15104025, "Copper"),
    (151040, "Metals & Mining", 15104030, "Gold"),
    (151040, "Metals & Mining", 15104040, "Precious Metals & Minerals"),
    (151040, "Metals & Mining", 15104045, "Silver"),
    (151040, "Metals & Mining", 15104050, "Steel"),
    (151050, "Paper & Forest Products", 15105010, "Forest Products"),
    (151050, "Paper & Forest Products", 15105020, "Paper Products"),
]

# Industrials classification
industrials_data = [
    (20, "Industrials", 2010, "Capital Goods"),
    (201010, "Aerospace & Defense", 20101010, "Aerospace & Defense"),
    (201020, "Building Products", 20102010, "Building Products"),
    (201030, "Construction & Engineering", 20103010, "Construction & Engineering"),
    (201040, "Electrical Equipment", 20104010, "Electrical Components & Equipment"),
    (201040, "Electrical Equipment", 20104020, "Heavy Electrical Equipment"),
    (201050, "Industrial Conglomerates", 20105010, "Industrial Conglomerates"),
    (201060, "Machinery", 20106010, "Construction Machinery & Heavy Transportation Equipment"),
    (201060, "Machinery", 20106015, "Agricultural & Farm Machinery"),
    (201060, "Machinery", 20106020, "Industrial Machinery & Supplies & Components"),
    (201070, "Trading Companies & Distributors", 20107010, "Trading Companies & Distributors"),
    (2020, "Commercial & Professional Services", 202010, "Commercial Services & Supplies"),
    (202010, "Commercial Services & Supplies", 20201010, "Commercial Printing"),
    (202010, "Commercial Services & Supplies", 20201050, "Environmental & Facilities Services"),
    (202010, "Commercial Services & Supplies", 20201060, "Office Services & Supplies"),
    (202010, "Commercial Services & Supplies", 20201070, "Diversified Support Services"),
    (202010, "Commercial Services & Supplies", 20201080, "Security & Alarm Services"),
    (202020, "Professional Services", 20202010, "Human Resource & Employment Services"),
    (202020, "Professional Services", 20202020, "Research & Consulting Services"),
    (202020, "Professional Services", 20202030, "Data Processing & Outsourced Services"),
    (2030, "Transportation", 203010, "Air Freight & Logistics"),
    (203010, "Air Freight & Logistics", 20301010, "Air Freight & Logistics"),
    (203020, "Passenger Airlines", 20302010, "Passenger Airlines"),
    (203030, "Marine Transportation", 20303010, "Marine Transportation"),
    (203040, "Ground Transportation", 20304010, "Rail Transportation"),
    (203040, "Ground Transportation", 20304030, "Cargo Ground Transportation"),
    (203040, "Ground Transportation", 20304040, "Passenger Ground Transportation"),
    (203050, "Transportation Infrastructure", 20305010, "Airport Services"),
    (203050, "Transportation Infrastructure", 20305020, "Highways & Railtracks"),
    (203050, "Transportation Infrastructure", 20305030, "Marine Ports & Services"),
]

table1_data = energy_data + materials_data + industrials_data
df1 = pd.DataFrame(table1_data, columns=["Sector", "Industry Group", "Industry", "Sub-Industry"])

# ---------------------------
# Table 2: Consumer Discretionary, Consumer Staples, Health Care
# ---------------------------
# Consumer Discretionary classification
consumer_discretionary_data = [
    (25, "Consumer Discretionary", 2510, "Automobiles & Components"),
    (251010, "Automobile Components", 25101010, "Automotive Parts & Equipment"),
    (251010, "Automobile Components", 25101020, "Tires & Rubber"),
    (251020, "Automobiles", 25102010, "Automobile Manufacturers"),
    (251020, "Automobiles", 25102020, "Motorcycle Manufacturers"),
    (2520, "Consumer Durables & Apparel", 252010, "Household Durables"),
    (252010, "Household Durables", 25201010, "Consumer Electronics"),
    (252010, "Household Durables", 25201020, "Home Furnishings"),
    (252010, "Household Durables", 25201030, "Homebuilding"),
    (252010, "Household Durables", 25201040, "Household Appliances"),
    (252010, "Household Durables", 25201050, "Housewares & Specialties"),
    (252020, "Leisure Products", 25202010, "Leisure Products"),
    (252030, "Textiles, Apparel & Luxury Goods", 25203010, "Apparel, Accessories & Luxury Goods"),
    (252030, "Textiles, Apparel & Luxury Goods", 25203020, "Footwear"),
    (252030, "Textiles, Apparel & Luxury Goods", 25203030, "Textiles"),
    (2530, "Consumer Services", 253010, "Hotels, Restaurants & Leisure"),
    (253010, "Hotels, Restaurants & Leisure", 25301010, "Casinos & Gaming"),
    (253010, "Hotels, Restaurants & Leisure", 25301020, "Hotels, Resorts & Cruise Lines"),
    (253010, "Hotels, Restaurants & Leisure", 25301030, "Leisure Facilities"),
    (253010, "Hotels, Restaurants & Leisure", 25301040, "Restaurants"),
    (253020, "Diversified Consumer Services", 25302010, "Education Services"),
    (253020, "Diversified Consumer Services", 25302020, "Specialized Consumer Services"),
    (2550, "Consumer Discretionary Distribution & Retail", 255010, "Distributors"),
    (255010, "Distributors", 25501010, "Distributors"),
    (255030, "Broadline Retail", 25503030, "Broadline Retail"),
    (255040, "Specialty Retail", 25504010, "Apparel Retail"),
    (255040, "Specialty Retail", 25504020, "Computer & Electronics Retail"),
    (255040, "Specialty Retail", 25504030, "Home Improvement Retail"),
    (255040, "Specialty Retail", 25504040, "Other Specialty Retail"),
    (255040, "Specialty Retail", 25504050, "Automotive Retail"),
    (255040, "Specialty Retail", 25504060, "Homefurnishing Retail"),
]

# Consumer Staples classification
consumer_staples_data = [
    (30, "Consumer Staples", 3010, "Consumer Staples Distribution & Retail"),
    (301010, "Consumer Staples Distribution & Retail", 30101010, "Drug Retail"),
    (301010, "Consumer Staples Distribution & Retail", 30101020, "Food Distributors"),
    (301010, "Consumer Staples Distribution & Retail", 30101030, "Food Retail"),
    (301010, "Consumer Staples Distribution & Retail", 30101040, "Consumer Staples Merchandise Retail"),
    (3020, "Food, Beverage & Tobacco", 302010, "Beverages"),
    (302010, "Beverages", 30201010, "Brewers"),
    (302010, "Beverages", 30201020, "Distillers & Vintners"),
    (302010, "Beverages", 30201030, "Soft Drinks & Non-alcoholic Beverages"),
    (302020, "Food Products", 30202010, "Agricultural Products & Services"),
    (302020, "Food Products", 30202030, "Packaged Foods & Meats"),
    (302030, "Tobacco", 30203010, "Tobacco"),
    (3030, "Household & Personal Products", 303010, "Household Products"),
    (303010, "Household Products", 30301010, "Household Products"),
    (303020, "Household & Personal Products", 30302010, "Personal Care Products"),
]

# Health Care classification
health_care_data = [
    (35, "Health Care", 3510, "Health Care Equipment & Services"),
    (351010, "Health Care Equipment & Supplies", 35101010, "Health Care Equipment"),
    (351010, "Health Care Equipment & Supplies", 35101020, "Health Care Supplies"),
    (351020, "Health Care Providers & Services", 35102010, "Health Care Distributors"),
    (351020, "Health Care Providers & Services", 35102015, "Health Care Services"),
    (351020, "Health Care Providers & Services", 35102020, "Health Care Facilities"),
    (351020, "Health Care Providers & Services", 35102030, "Managed Health Care"),
    (351030, "Health Care Technology", 35103010, "Health Care Technology"),
    (3520, "Pharmaceuticals, Biotechnology & Life Sciences", 352010, "Biotechnology"),
    (352010, "Biotechnology", 35201010, "Biotechnology"),
    (352020, "Pharmaceuticals", 35202010, "Pharmaceuticals"),
    (352030, "Life Sciences Tools & Services", 35203010, "Life Sciences Tools & Services"),
]

table2_data = consumer_discretionary_data + consumer_staples_data + health_care_data
df2 = pd.DataFrame(table2_data, columns=["Sector", "Industry Group", "Industry", "Sub-Industry"])

# ---------------------------
# Table 3: Financials, Information Technology, Communication Services, Utilities, Real Estate
# ---------------------------
# Financials classification
financials_data = [
    (40, "Financials", 4010, "Banks"),
    (401010, "Banks", 40101010, "Diversified Banks"),
    (401010, "Banks", 40101015, "Regional Banks"),
    (4020, "Financial Services", 402010, "Financial Services"),
    (402010, "Financial Services", 40201020, "Diversified Financial Services"),
    (402010, "Financial Services", 40201030, "Multi-Sector Holdings"),
    (402010, "Financial Services", 40201040, "Specialized Finance"),
    (402010, "Financial Services", 40201050, "Commercial & Residential Mortgage Finance"),
    (402010, "Financial Services", 40201060, "Transaction & Payment Processing Services"),
    (402020, "Consumer Finance", 40202010, "Consumer Finance"),
    (402030, "Capital Markets", 40203010, "Asset Management & Custody Banks"),
    (402030, "Capital Markets", 40203020, "Investment Banking & Brokerage"),
    (402030, "Capital Markets", 40203030, "Diversified Capital Markets"),
    (402030, "Capital Markets", 40203040, "Financial Exchanges & Data"),
    (402040, "Mortgage Real Estate Investment Trusts (REITs)", 40204010, "Mortgage REITs"),
    (4030, "Insurance", 403010, "Insurance"),
    (403010, "Insurance", 40301010, "Insurance Broker"),
    (403010, "Insurance", 40301020, "Life & Health Insurance"),
    (403010, "Insurance", 40301030, "Multi-line Insurance"),
    (403010, "Insurance", 40301040, "Property & Casualty Insurance"),
    (403010, "Insurance", 40301050, "Reinsurance"),
]

# Information Technology classification
information_technology_data = [
    (45, "Information Technology", 4510, "Software & Services"),
    (451020, "IT Services", 45102010, "IT Consulting & Other Services"),
    (451020, "IT Services", 45102030, "Internet Services & Infrastructure"),
    (451030, "Software", 45103010, "Application Software"),
    (451030, "Software", 45103020, "Systems Software"),
    (4520, "Technology Hardware & Equipment", 452010, "Communications Equipment"),
    (452010, "Communications Equipment", 45201020, "Communications Equipment"),
    (452020, "Technology Hardware, Storage & Peripherals", 45202030, "Technology Hardware, Storage & Peripherals"),
    (452030, "Electronic Equipment, Instruments & Components", 45203010, "Electronic Equipment & Instruments"),
    (452030, "Electronic Equipment, Instruments & Components", 45203015, "Electronic Components"),
    (452030, "Electronic Equipment, Instruments & Components", 45203020, "Electronic Manufacturing Services"),
    (452030, "Electronic Equipment, Instruments & Components", 45203030, "Technology Distributors"),
    (4530, "Semiconductors & Semiconductor Equipment", 453010, "Semiconductors & Semiconductor Equipment"),
    (453010, "Semiconductors & Semiconductor Equipment", 45301010, "Semiconductor Materials & Equipment"),
    (453010, "Semiconductors & Semiconductor Equipment", 45301020, "Semiconductors"),
]

# Communication Services classification
communication_services_data = [
    (50, "Communication Services", 5010, "Telecommunication Services"),
    (501010, "Diversified Telecommunication Services", 50101010, "Alternative Carriers"),
    (501010, "Diversified Telecommunication Services", 50101020, "Integrated Telecommunication Services"),
    (501020, "Wireless Telecommunication Services", 50102010, "Wireless Telecommunication Services"),
    (5020, "Media & Entertainment", 502010, "Media"),
    (502010, "Media", 50201010, "Advertising"),
    (502010, "Media", 50201020, "Broadcasting"),
    (502010, "Media", 50201030, "Cable & Satellite"),
    (502010, "Media", 50201040, "Publishing"),
    (502020, "Entertainment", 50202010, "Movies & Entertainment"),
    (502020, "Entertainment", 50202020, "Interactive Home Entertainment"),
    (502030, "Interactive Media & Services", 50203010, "Interactive Media & Services"),
]

# Utilities classification
utilities_data = [
    (55, "Utilities", 5510, "Utilities"),
    (551010, "Electric Utilities", 55101010, "Electric Utilities"),
    (551020, "Gas Utilities", 55102010, "Gas Utilities"),
    (551030, "Multi-Utilities", 55103010, "Multi-Utilities"),
    (551040, "Water Utilities", 55104010, "Water Utilities"),
    (551050, "Independent Power and Renewable Electricity Producers", 55105010, "Independent Power Producers & Energy Traders"),
    (551050, "Independent Power and Renewable Electricity Producers", 55105020, "Renewable Electricity"),
]

# Real Estate classification
real_estate_data = [
    (60, "Real Estate", 6010, "Equity Real Estate Investment Trusts (REITs)"),
    (601010, "Equity Real Estate Investment Trusts (REITs)", 60101010, "Diversified REITs"),
    (601025, "Industrial REITs", 60102510, "Industrial REITs"),
    (601030, "Hotel & Resort REITs", 60103010, "Hotel & Resort REITs"),
    (601040, "Office REITs", 60104010, "Office REITs"),
    (601050, "Health Care REITs", 60105010, "Health Care REITs"),
    (601060, "Residential REITs", 60106010, "Multi-Family Residential REITs"),
    (601060, "Residential REITs", 60106020, "Single-Family Residential REITs"),
    (601070, "Retail REITs", 60107010, "Retail REITs"),
    (601080, "Specialized REITs", 60108010, "Other Specialized REITs"),
    (601080, "Specialized REITs", 60108020, "Self-Storage REITs"),
    (601080, "Specialized REITs", 60108030, "Telecom Tower REITs"),
    (601080, "Specialized REITs", 60108040, "Timber REITs"),
    (601080, "Specialized REITs", 60108050, "Data Center REITs"),
    (6020, "Real Estate Management & Development", 602010, "Real Estate Management & Development"),
    (602010, "Real Estate Management & Development", 60201010, "Diversified Real Estate Activities"),
    (602010, "Real Estate Management & Development", 60201020, "Real Estate Operating Companies"),
    (602010, "Real Estate Management & Development", 60201030, "Real Estate Development"),
    (602010, "Real Estate Management & Development", 60201040, "Real Estate Services"),
]

table3_data = (financials_data + information_technology_data +
               communication_services_data + utilities_data + real_estate_data)
df3 = pd.DataFrame(table3_data, columns=["Sector", "Industry Group", "Industry", "Sub-Industry"])

# ---------------------------
# Combine all tables into one DataFrame
# ---------------------------
df_all = pd.concat([df1, df2, df3], ignore_index=True)

# Helper: count number of digits in a numeric code
num_digits = lambda x: len(str(x))

# ---------------------------
# Create the three dictionaries
# ---------------------------
# 1. Sector Dictionary: rows where the 'Sector' code is 2 digits
sector_df = df_all[df_all["Sector"].apply(lambda x: num_digits(x) <= 2)].drop_duplicates(subset=["Sector", "Industry Group"])
sector_dict = dict(zip(sector_df["Sector"], sector_df["Industry Group"]))

# 2. Industry Dictionary: rows where the 'Industry' code is 4 digits (aggregated rows)
industry_df = df_all[df_all["Industry"].apply(lambda x: num_digits(x) == 4)].drop_duplicates(subset=["Industry"])
industry_dict = dict(zip(industry_df["Industry"], industry_df["Sub-Industry"]))

# 3. Sub-Industry Dictionary: rows where the 'Industry' code is 8 digits (detailed rows)
subindustry_df = df_all[df_all["Industry"].apply(lambda x: num_digits(x) == 8)].drop_duplicates(subset=["Industry"])
subindustry_dict = dict(zip(subindustry_df["Industry"], subindustry_df["Sub-Industry"]))

# ---------------------------
# Output the three dictionaries
# ---------------------------
print("Sector Dictionary:")
print(sector_dict)

print("\nIndustry Dictionary:")
print(industry_dict)

print("\nSub-Industry Dictionary:")
print(subindustry_dict)
