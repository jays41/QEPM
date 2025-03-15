# Market Neutral QEPM Model

ctrl k, v <------------------------------------------

### Data Processing & Risk Modeling
- Processes historical stock price data with handling of outliers and missing values
- Calculates expected returns using exponentially weighted moving averages
- Constructs covariance matrices with regularisation techniques (regularised by adding a small multiple of the identity matrix when the minimum eigenvalue is too small, ensuring it remains well-conditioned)
- Computes beta factors for market neutrality constraints (Betas are calculated as the covariance of each stock's returns with the market returns (S&P 500) divided by the market variance, with extreme values clipped between -3.0 and 3.0)

### Portfolio Optimization
- Implements a convex optimization approach using multiple solvers (SCS, OSQP, ECOS, CVXOPT)
- Enforces critical constraints:
  - Dollar neutrality (equal long and short positions)
  - Beta neutrality (market factor neutrality)
  - Sector neutrality within defined risk parameters
  - Position size limits (±10% per position)
  - Overall risk target adherence

### Backtesting & Analysis
- Tests portfolio performance using historical data
- Calculates key metrics: annualised returns, volatility, Sharpe ratio
- Visualizes risk-return relationship across various target risk levels

## Results
The system successfully generates portfolios across a range of annual risk targets. Performance analysis demonstrates a clear relationship between target risk levels and achieved returns, with most optimisations reaching optimal solutions.





It is not a fundamental or economic model **yet**: just need to change the way that expected returns are calculated to inlcude the factors rather than calculating returns purely from statistical data


To implement:
    Factor selection
    Factor scoring (aggregate Z-scores) (partially done)
    Calculate expected returns:
        E[R_i] = α_i + β_i,1 × F_1 + β_i,2 × F_2 + ... + β_i,n × F_n
        -> E[R_i] is the expected return for stock i
        -> α_i is the stock-specific component
        -> β_i,j is the exposure (sensitivity) of stock i to factor j
        -> F_j is the expected return of factor j
    Can add factors as an additional constraint