import pandas as pd
import matplotlib.pyplot as plt
import os

# =============================================================================
# 1. LOAD BACKTEST RESULTS
# =============================================================================
df_results = pd.read_csv('backtest_results/pooled_ols_and_spearman_results.csv')
# Must have columns like: year, gvkey, commodity, alpha, beta, spearman_corr, r_squared, etc.

print("\nLoaded results. Columns:\n", df_results.columns)

# =============================================================================
# 2. MERGE IN SECTOR/INDUSTRY/SUBINDUSTRY CLASSIFICATION
# =============================================================================
# Load the classification table (repeated rows per gvkey).
df_class = pd.read_csv('data/Company Sector Translation Table.csv')
# Typically has columns like: gvkey, datadate, gsector, gind, gsubind, etc.

# Sort so earliest date is first, then group by gvkey
df_class_sorted = df_class.sort_values(['gvkey','datadate'])
df_class_first = df_class_sorted.groupby('gvkey', as_index=False).first()

# Keep only the columns we need (gsector, gind, gsubind)
# If your file has differently named columns, adjust accordingly
df_class_first = df_class_first[['gvkey','gsector','gind','gsubind']]

# Merge classification onto the results, so each row gains a stable sector/industry/subindustry
df_merged = pd.merge(
    df_results,
    df_class_first,  # already limited to 1 row per gvkey
    on='gvkey',
    how='left'
)

print("After merging classification, columns:\n", df_merged.columns)

# =============================================================================
# 3. HELPER FUNCTION FOR CLASSIFICATION ANALYSIS
# =============================================================================
def analyze_by_classification(df, class_col, folder_name, class_label=""):
    """
    df         : DataFrame with columns [year, spearman_corr, commodity, ... , class_col]
    class_col  : The classification column name (e.g. 'gsector', 'gind', 'gsubind')
    folder_name: Subfolder name for output (will be created under backtest_results/)
    class_label: For chart titles (e.g. "Sector", "Industry", "Subindustry")
    """
    outdir = os.path.join('backtest_results', folder_name)
    os.makedirs(outdir, exist_ok=True)

    # -------------------------------------------------------------------------
    # A) Summaries of correlation, aggregated across all years & commodities
    # -------------------------------------------------------------------------
    df_class_summary = (
        df.groupby(class_col, dropna=False)['spearman_corr']
          .mean()
          .reset_index(name='avg_spearman_corr')
          .sort_values('avg_spearman_corr', ascending=False)
    )

    print(f"\nSummary by {class_label} (all years & commodities):")
    print(df_class_summary)

    # 1) Bar Chart: average correlation by this classification (all years & commodities)
    plt.figure()
    plt.bar(df_class_summary[class_col].astype(str), df_class_summary['avg_spearman_corr'])
    plt.title(f"Avg Spearman Corr by {class_label}\n(All Years & Commodities)")
    plt.xlabel(class_label)
    plt.ylabel("Avg Spearman Corr")
    plt.xticks(rotation=90)
    plt.tight_layout()
    fname1 = os.path.join(outdir, f"avg_corr_by_{class_col}_all_years.png")
    plt.savefig(fname1, dpi=300)
    plt.close()
    print(f"Saved bar chart: {fname1}")

    # 2) Line Chart: correlation by year, grouping by classification (aggregating all commodities)
    df_class_year = (
        df.groupby(['year', class_col], dropna=False)['spearman_corr']
          .mean()
          .reset_index(name='avg_spearman_corr')
          .sort_values(['year', class_col])
    )
    df_class_year_pivot = df_class_year.pivot(index='year', columns=class_col, values='avg_spearman_corr')

    plt.figure()
    for cval in df_class_year_pivot.columns:
        plt.plot(df_class_year_pivot.index, df_class_year_pivot[cval], label=str(cval))

    plt.title(f"Average Spearman Corr by Year & {class_label}\n(All Commodities Aggregated)")
    plt.xlabel("Year")
    plt.ylabel("Avg Spearman Corr")
    plt.legend(title=class_label, loc='best', fontsize='small')
    plt.tight_layout()
    fname2 = os.path.join(outdir, f"corr_over_time_by_{class_col}.png")
    plt.savefig(fname2, dpi=300)
    plt.close()
    print(f"Saved line chart: {fname2}")

    # -------------------------------------------------------------------------
    # 3) (Optional) For each commodity, create a bar chart of correlation 
    #    by classification. This can produce MANY charts if you have many commodities.
    # -------------------------------------------------------------------------
    df_class_commodity = (
        df.groupby(['commodity', class_col], dropna=False)['spearman_corr']
          .mean()
          .reset_index(name='avg_spearman_corr')
    )
    all_commodities = df_class_commodity['commodity'].dropna().unique()
    all_commodities.sort()

    subfolder_commodity = os.path.join(outdir, f"{class_col}_commodity_charts")
    os.makedirs(subfolder_commodity, exist_ok=True)

    for comm in all_commodities:
        df_sub = df_class_commodity[df_class_commodity['commodity'] == comm].copy()
        if df_sub.empty:
            continue

        # Sort by correlation descending
        df_sub.sort_values('avg_spearman_corr', ascending=False, inplace=True)

        plt.figure()
        plt.bar(df_sub[class_col].astype(str), df_sub['avg_spearman_corr'])
        plt.title(f"Avg Corr by {class_label}\nCommodity = {comm}")
        plt.xlabel(class_label)
        plt.ylabel("Avg Spearman Corr")
        plt.xticks(rotation=90)
        plt.tight_layout()

        fname3 = os.path.join(subfolder_commodity, f"avg_corr_by_{class_col}_{comm.replace(' ','_')}.png")
        plt.savefig(fname3, dpi=300)
        plt.close()

        print(f"Saved chart: {fname3}")

# =============================================================================
# 4. PERFORM THE ANALYSIS FOR gsector, gind, gsubind
# =============================================================================
analyze_by_classification(
    df=df_merged,
    class_col='gsector',
    folder_name='by_sector_charts',
    class_label="Sector"
)

analyze_by_classification(
    df=df_merged,
    class_col='gind',
    folder_name='by_industry_charts',
    class_label="Industry"
)

analyze_by_classification(
    df=df_merged,
    class_col='gsubind',
    folder_name='by_subindustry_charts',
    class_label="Subindustry"
)

print("\nAll sector/industry/subindustry analyses & plots are saved under 'backtest_results'!")
