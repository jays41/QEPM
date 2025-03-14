p/b
ev/ebitda
free cash flow yield
pegy
revenue growth
roce
interest coverage ratio
asset turnover ratio
dividend yield
dividend growth
market cap -> from daily_stock_price_data.csv # TODO <--------------------------------------------------
momentum -> calculate manually

find mean and std for each factor for each stock per yearly


fun zScore (start date, end date, stock list, factor list) {
    overall_z_scores = []
    for stock in stock list {
        z = 0
        for factor in factors {
            lastValue, mean, std = getFactorData(start date, end date, stock, factor)
            z += (lastValue - mean) / std
        }
        # TODO: add weighting here -> from Tristan's regressions
        overall_z_scores.append((z, stock, sector))
    }
    overall_z_scores.sort()
    top = overall_z_scores[~top10%]
    bottom = overall_z_scores[~bottom10%]
}

fun getFactorData (start date, end date, stock, factor) {
    get data for the factor for the year
    filter by stock
    calculate mean and std for the last year
    return lastValue, mean, std
}


method:
map econ factors to sectors for whole universe

get z scores for each stock for our list of factors
take top and bottom 10% of these
for the stocks we have left, split into sectors