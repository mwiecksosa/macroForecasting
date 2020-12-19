import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARMAResults

import statsmodels.api as sm
import matplotlib.pyplot as plt

rootPath = "/home/mwiecksosa/stat429/"

def main():
    filename = rootPath+"m-ew6299.txt"
    df = pd.DataFrame(np.loadtxt(filename).reshape(-1, 1))
    X = df[0]

    # AR(3)
    mod = ARMA(X, order=(3,0))
    res = mod.fit()
    print(res.summary())
    print(res.predict())
    print(res.predict(456,456))
    print(res.predict(456,457))

    # MA(3)
    mod = ARMA(X, order=(0,3))
    res = mod.fit()
    print(res.summary())
    print(res.predict())
    print(res.predict(456,456))
    print(res.predict(456,457))

if __name__ == '__main__':
    main()
