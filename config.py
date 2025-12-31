from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    DATA_PATH: str = "QEPM/data/"
    RESULTS_PATH: str = "QEPM/results/"
    
    STOCK_PRICES_FILE: str = "stock_prices.csv"
    ALL_DATA_FILE: str = "all_data.csv"
    FUNDAMENTAL_DATA_FILE: str = "stock_fundamental_data.csv"
    TECHNICAL_DATA_FILE: str = "technical_factors.csv"
    SECTOR_DATA_FILE: str = "Company Sector Translation Table.csv"
    

    FUNDAMENTAL_FACTORS: List[str] = field(default_factory=lambda: [
        'npm',              # Net Profit Margin
        'opmad',            # Operating Margin (adjusted)
        'gpm',              # Gross Profit Margin
        'ptpm',             # Pretax Profit Margin
        'pretret_earnat',   # Pretax Return Earnings NAT
        'equity_invcap',    # Equity to Invested Capital
        'debt_invcap',      # Debt to Invested Capital
        'capital_ratio',    # Capital Ratio
        'invt_act',         # Inventory to Assets
        'rect_act',         # Receivables to Assets
        'debt_assets',      # Debt to Assets
        'debt_capital',     # Debt to Capital
        'cash_ratio',       # Cash Ratio
        'adv_sale'          # Advertising to Sales
    ])
    
    TECHNICAL_FACTORS: List[str] = field(default_factory=lambda: [
        'macd_30'
    ])
    
    WINSORIZE_PERCENTILE: float = 0.05      # Cap extreme values at 5th/95th percentile
    TOP_PERCENTILE: float = 0.20            # Select top 20% of stocks by composite Z-score
    MIN_STOCKS: int = 30                     # Minimum stocks required to proceed
    
    RETURNS_FREQ: str = 'M'                  # 'M' for monthly, 'Q' for quarterly, 'B' for biannually, 'A' for annually
    PERIODS_PER_YEAR: int = 12               # 12 for monthly, 4 for quarterly
    LOOKBACK_DAYS: int = 252                 # Trading days for historical calculations
    
    TARGET_ANNUAL_RISK: float = 0.15         # Target portfolio volatility (15%)
    COV_LOOKBACK_PERIODS: int = 60           # Months of data for covariance matrix
    SHRINKAGE_INTENSITY: float = 0.1         # Ledoit-Wolf shrinkage (0 = none, 1 = full)
    
    MAX_POSITION_SIZE: float = 0.10          # Max 10% in any single stock
    MIN_POSITION_SIZE: float = 0.005         # Min 0.5% position (avoid dust)
    MAX_SECTOR_WEIGHT: float = 0.30          # Max 30% in any sector
    MAX_TURNOVER: float = 0.50               # Max 50% turnover per rebalance
    
    REBALANCE_FREQ: str = 'Q'                # 'M' monthly, 'Q' quarterly, 'A' annual
    START_DATE: str = '2015-01-01'
    END_DATE: str = '2023-12-31'
    INITIAL_CAPITAL: float = 1_000_000       # $1M starting capital
    
    TAU_MODE: str = 'off'                    # 'off' | 'avg' | 'stock'
    TAU_SCALE: float = 0.05                  # Scaling factor for tau adjustments
    TAU_FACTORS: List[str] = field(default_factory=lambda: ['macd_30'])
    
    SAVE_RESULTS: bool = True
    VERBOSE: bool = True
    PLOT_RESULTS: bool = False
    
    def get_full_path(self, filename: str) -> str:
        return f"{self.DATA_PATH}{filename}"
    
    def get_all_factors(self) -> List[str]:
        return self.FUNDAMENTAL_FACTORS + self.TECHNICAL_FACTORS
    
    def __post_init__(self):
        if self.RETURNS_FREQ == 'M':
            self.PERIODS_PER_YEAR = 12
        elif self.RETURNS_FREQ == 'Q':
            self.PERIODS_PER_YEAR = 4
        elif self.RETURNS_FREQ == 'B':
            self.PERIODS_PER_YEAR = 2
        elif self.RETURNS_FREQ == 'A':
            self.PERIODS_PER_YEAR = 1
        else:
            raise ValueError(f"Invalid RETURNS_FREQ: {self.RETURNS_FREQ}")

CONFIG = Config()