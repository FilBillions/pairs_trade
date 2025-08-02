from statsmodels.tsa.stattools import coint, adfuller

def check_for_stationarity(x, cutoff=0.05, print_results=True):
    #H_0 in adfuller is the data series is non-stationary
    # If pvalue < cutoff then data is likely stationary
    pvalue = adfuller(x)[1]
    if pvalue < cutoff:
        if print_results:
            print(f"p-value = {pvalue} | The series {x.name} is likely stationary")
        return True
    else:
        if print_results:
            print(f"p-value = {pvalue} | The series {x.name} is likely non-stationary")
        return False