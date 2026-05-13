import numpy as np

def lgamma(x):
    """
    Computes log(Gamma(x)) for integer and half-integer values using numpy.
    Since we only need it for n/2 + 1 (where n is an integer), 
    x will always be either an integer or a half-integer.
    """
    # If x is a whole number (integer)
    if x % 1 == 0:
        k = int(x)
        if k <= 1:
            return 0.0
        # log(Gamma(k)) = log((k-1)!) = sum(log(i)) for i = 1 to k-1
        return np.sum(np.log(np.arange(1, k)))
    else:
        # If x is a half-integer (e.g., 1.5, 2.5)
        # Gamma(k + 0.5) = (k - 0.5) * (k - 1.5) * ... * 0.5 * sqrt(pi)
        k = int(np.floor(x))
        if k == 0:
            return np.log(np.sqrt(np.pi))
        terms = np.arange(0.5, k + 0.5, 1.0)
        return np.sum(np.log(terms)) + np.log(np.sqrt(np.pi))

def calculate_vol(n, r):
    """
    Calculate the volume of an n-dimensional ball of radius r.
    Uses logarithmic properties to handle large values of n without overflow.
    """
    # Volume of n-ball: V_n(R) = (pi^(n/2) / Gamma(n/2 + 1)) * R^n
    # Taking log: log(V_n) = (n/2)*log(np.pi) - log(Gamma(n/2 + 1)) + n*log(R)
    log_vol = (n / 2) * np.log(np.pi) - lgamma((n / 2) + 1) + n * np.log(r)
    return np.exp(log_vol)

if __name__ == "__main__":
    # Test cases
    radius = 1.0
    print(f"Testing volume of n-ball with radius {radius}:\n")
    
    # n=2: Circle -> Area = pi * r^2
    vol_2d = calculate_vol(2, radius)
    print(f"n=2 (Circle): {vol_2d:.4f} (Expected: {np.pi:.4f})")
    
    # n=3: Sphere -> Volume = 4/3 * pi * r^3
    vol_3d = calculate_vol(3, radius)
    expected_3d = (4/3) * np.pi * (radius**3)
    print(f"n=3 (Sphere): {vol_3d:.4f} (Expected: {expected_3d:.4f})")
    
    # n=4: 4-Ball -> Volume = 1/2 * pi^2 * r^4
    vol_4d = calculate_vol(4, radius)
    expected_4d = 0.5 * (np.pi**2) * (radius**4)
    print(f"n=4 (4-Ball): {vol_4d:.4f} (Expected: {expected_4d:.4f})")
    
    print("\nVolume starts decreasing for larger dimensions:")
    for n in [5, 10, 20]:
        print(f"n={n}: {calculate_vol(n, radius):.6e}")