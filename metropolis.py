import numpy as np

# Metropolis sampler with symmetric proposal function
def metropolis(num_samples, initial_x, target_fn, proposal_fn, stepsize):
    samples = [initial_x]
    x = initial_x
    accepted = 0  # counter for accepted proposals

    for _ in range(num_samples - 1):
        # Pass the stepsize into the proposal function
        x_proposal = proposal_fn(x, stepsize)
        acceptance_ratio = target_fn(x_proposal) / target_fn(x)

        if np.random.rand() < min(1, acceptance_ratio):
            x = x_proposal
            accepted += 1

        samples.append(x)

    acceptance_rate = accepted / (num_samples - 1)
    
    # Returns both the samples array and the acceptance rate
    return np.array(samples), acceptance_rate