import numpy as np                         # numpy namespace
from timeit import default_timer as timer  # for timing
from matplotlib import pyplot              # for plotting
import math
from numba import vectorize, cuda, jit
import accelerate.cuda

@jit('void(double[:], double[:], double, double, double, double[:])',target='gpu')
def step_cuda(last, paths, dt, c0, c1, normdist):
    i = cuda.grid(1)
    if i >= paths.shape[0]:
        return
    noise = normdist[i]
    paths[i] = last[i] * math.exp(c0 * dt + c1 * noise)

def mc_cuda(paths, dt, interest, volatility):
    n = paths.shape[0]

    blksz = cuda.get_current_device().MAX_THREADS_PER_BLOCK
    gridsz = int(math.ceil(float(n) / blksz))

    # instantiate a CUDA stream for queueing async CUDA cmds
    stream = cuda.stream()
    # instantiate a cuRAND PRNG
    prng = accelerate.cuda.rand.PRNG(accelerate.cuda.rand.PRNG.MRG32K3A, stream=stream)

    # Allocate device side array
    d_normdist = cuda.device_array(n, dtype=np.double, stream=stream)
    
    c0 = interest - 0.5 * volatility ** 2
    c1 = volatility * math.sqrt(dt)

    # configure the kernel
    # similar to CUDA-C: step_cuda<<<gridsz, blksz, 0, stream>>>
    step_cfg = step_cuda[gridsz, blksz, stream]
    
    # transfer the initial prices
    d_last = cuda.to_device(paths[:, 0], stream=stream)
    for j in range(1, paths.shape[1]):
        # call cuRAND to populate d_normdist with gaussian noises
        prng.normal(d_normdist, mean=0, sigma=1)
        # setup memory for new prices
        # device_array_like is like empty_like for GPU
        d_paths = cuda.device_array_like(paths[:, j], stream=stream)
        # invoke step kernel asynchronously
        step_cfg(d_last, d_paths, dt, c0, c1, d_normdist)
        # transfer memory back to the host
        d_paths.copy_to_host(paths[:, j], stream=stream)
        d_last = d_paths
    # wait for all GPU work to complete
    stream.synchronize()

@vectorize(['f8(f8, f8, f8, f8, f8)'],target='cuda')
def step_gpuvec(last, dt, c0, c1, noise):
    return last * math.exp(c0 * dt + c1 * noise)

def mc_gpuvec(paths, dt, interest, volatility):
    c0=interest-0.5*volatility**2
    c1=volatility*np.sqrt(dt)

    for j in range(1, paths.shape[1]):
        prices = paths[:,j-1]
        noises = np.random.normal(0., 1., prices.size)
        paths[:, j] = step_gpuvec(prices, dt, c0, c1, noises)

@vectorize(['f8(f8, f8, f8, f8, f8)'],target='parallel')
def step_parallel(last, dt, c0, c1, noise):
    return last * math.exp(c0 * dt + c1 * noise)

def mc_parallel(paths, dt, interest, volatility):
    c0=interest-0.5*volatility**2
    c1=volatility*np.sqrt(dt)

    for j in range(1, paths.shape[1]):
        prices = paths[:,j-1]
        noises = np.random.normal(0., 1., prices.size)
        paths[:, j] = step_parallel(prices, dt, c0, c1, noises)

@vectorize(['f8(f8, f8, f8, f8, f8)'])
def step_cpuvec(last, dt, c0, c1, noise):
    return last * math.exp(c0 * dt + c1 * noise)

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
    print(fmt % ('standard error' , np.std(ST) / math.sqrt(NumPath)))
    print(fmt % ('paid off', np.mean(PaidOff)))
    optionprice = np.mean(PaidOff) * math.exp(-InterestRate * Maturity)
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
numpy_time = driver(mc_numpy, do_plot=False)
print('\nUsing CPU vectorization:')
cpuvec_time = driver(mc_cpuvec, do_plot=False)
print('\nUsing using parallel CPU threads')
parallel_time = driver(mc_parallel, do_plot=False)
print('\nUsing GPU vectorization')
gpuvec_time = driver(mc_gpuvec, do_plot=False)
print('\nUsing CUDA')
cuda_time = driver(mc_cuda, do_plot=False)

def perf_plot(rawdata, xlabels):
    data = [numpy_time / x for x in rawdata]
    idx = np.arange(len(data))
    fig = pyplot.figure()
    width = 0.5
    ax = fig.add_subplot(111)
    ax.bar(idx, data, width)
    ax.set_ylabel('normalized speedup')
    ax.set_xticks(idx + width / 2)
    ax.set_xticklabels(xlabels)
    ax.set_ylim(0.9)
    pyplot.show()

perf_plot([numpy_time, cpuvec_time, parallel_time, gpuvec_time,cuda_time],
          ['numpy', 'cpu-vect', 'parallel-vect', 'gpu-vect','cuda'])
