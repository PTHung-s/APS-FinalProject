import numpy as np
from plot import plot_result
from metropolis import metropolis

def target_fn(x):
    return np.exp(-x**2 / 2)

def proposal_fn(x, stepsize):
    return x + np.random.uniform(-stepsize, stepsize)

def run(n=10000, x0=0.0, stepsize=1.0):
    samples, acc_rate = metropolis(
        n,
        x0,
        target_fn,
        proposal_fn,
        stepsize
    )
    plot_result(samples, acc_rate)

if __name__ == "__main__":
    run(10000)