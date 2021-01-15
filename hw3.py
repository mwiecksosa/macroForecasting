import pandas as pd
import numpy as np

dataPath="/mnt/c/Users/mykul/Downloads/"
fname="d-ibmy98.csv"

df = pd.read_csv(dataPath+fname)
df = df[['X19980102','X0.009558']]

def mean(data):
    n = len(data)
    if n < 1:
        raise ValueError('Need at least one data point')
    return sum(data)/n

def _ss(data):
    m = mean(data)
    ss = sum((x-m)**2 for x in data)
    return ss

def stddev(data, d=0):
    n = len(data)
    if n < 2:
        raise ValueError('Need at least two data points')
    ss = _ss(data)
    v = ss/(n-d)
    return v**0.5

simpleReturns = df['X0.009558']
logReturns = np.log(simpleReturns+1)

srLogReturns = stddev(logReturns,d=1)
print("srLogReturns",srLogReturns)

sigmahatLogReturns = srLogReturns / (1**(1/2))
print("sigmahatLogReturns",sigmahatLogReturns)

rbarLogReturns = mean(logReturns)
print("rbarLogReturns",rbarLogReturns)

muLogReturns = rbarLogReturns + (sigmahatLogReturns**2)/2
print("muLogReturns",muLogReturns)
