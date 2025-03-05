# QEPM

Need to come back to the alpha calculation -> to do for market neutral including the benchmark
Think about how to implement tracking error -> we need the weights of the benchmark

To implement:
    Factor selection -> program into preweighting
    Factor scoring (aggregate Z-scores) (partially done)
    Calculate expected returns:
        E[R_i] = α_i + β_i,1 × F_1 + β_i,2 × F_2 + ... + β_i,n × F_n
        -> E[R_i] is the expected return for stock i
        -> α_i is the stock-specific component
        -> β_i,j is the exposure (sensitivity) of stock i to factor j
        -> F_j is the expected return of factor j
    Can add factors as an additional constraint


Alpha calculation for the portfolio
Performance attribution (??)
Table presentation:

    dont forget corr with S&P