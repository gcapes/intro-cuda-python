import numpy as np                         # numpy namespace
from timeit import default_timer as timer  # for timing
from matplotlib import pyplot              # for plotting
from math import sqrt, exp
from numba import vectorize

@vectorize(['f8(f8, f8, f8, f8, f8)'])
def step_cpuvec(last, dt, c0, c1, noise):
    return last * exp(c0 * dt + c1 * noise)

def mc_cpuvec(paths, dt, interest, volatility):
    c0 = interest - 0.5 * volatility ** 2
    c1 = volatility * np.sqrt(dt)

    for j in range(1, paths.shape[1]):
        prices = paths[:, j - 1]
        noises = np.random.normal(0., 1., prices.size)
        paths[:, j] = step_cpuvec(prices, dt, c0, c1, noises)

def step_numpy(dt, prices, c0, c1, noises):
    return prices * np.exp(c0 * dt + c1 * noises)

def mc_numpy(paths, dt, interest, volatility):
    c0 = interest - 0.5 * volatility ** 2
    c1 = volatility * np.sqrt(dt)

    for j in range(1, paths.shape[1]):   # for each time step
        prices = paths[:, j - 1]          # last prices
        # gaussian noises for simulation
        noises = np.random.normal(0., 1., prices.size)
        # simulate
        paths[:, j] = step_numpy(dt, prices, c0, c1, noises)

# Configurations
# stock parameter

StockPrice = 20.83
StrikePrice = 21.50
Volatility = 0.021
InterestRate = 0.20
Maturity = 5. / 12.

# monte-carlo parameter 

NumPath = 3000000
NumStep = 100

# plotting
MAX_PATH_IN_PLOT = 50

# Driver
# The driver measures the performance of the given pricer and plots the simulation paths.
def driver(pricer, do_plot=False):
    paths = np.zeros((NumPath, NumStep + 1), order='F')
    paths[:, 0] = StockPrice
    DT = Maturity / NumStep

    ts = timer()
    pricer(paths, DT, InterestRate, Volatility)
    te = timer()
    elapsed = te - ts

    ST = paths[:, -1]
    PaidOff = np.maximum(paths[:, -1] - StrikePrice, 0)
    print('Result')
    fmt = '%20s: %s'
    print(fmt % ('stock price', np.mean(ST)))
    print(fmt % ('standard error' , np.std(ST) / sqrt(NumPath)))
    print(fmt % ('paid off', np.mean(PaidOff)))
    optionprice = np.mean(PaidOff) * exp(-InterestRate * Maturity)
    print(fmt % ('option price', optionprice))

    print('Performance')
    NumCompute = NumPath * NumStep
    print('%20s: %.2f' % ('Mstep/second', (NumCompute / elapsed / 1e6)))
    print('%20s: %.3fs' % ('time elapsed', (te - ts)))

    if do_plot:
        pathct = min(NumPath, MAX_PATH_IN_PLOT)
        for i in range(pathct):
            pyplot.plot(paths[i])
        print ('Plotting %d/%d paths' % (pathct, NumPath))
        pyplot.show()
    return elapsed

print('\nUsing numpy:')
numpy_time = driver(mc_numpy, do_plot=True)
print('\nUsing CPU vectorization:')
cpuvec_time = driver(mc_cpuvec, do_plot=True)
