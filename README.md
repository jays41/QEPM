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

look into sector beta neutrality

make it rolling - do expected returns based on a 5 year look back period and weight and then run the backtest on the upcoming year and then move the whole window by one year and do the same thing over and over again

change beta calculation to just used the saved data in s&p_data.csv
    getting s&p data -> should just write a script to download the data and save it

with the rolling backtest, make it so that stocks are dropped if they don't contain data over the whole 5 year range being looked at