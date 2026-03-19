"""
Statistics Notes
---------------
- Mean: Average value of a dataset.
- Variance: Measure of spread in the data.
- Standard Deviation: Square root of variance.
- Normal Distribution: Bell-shaped, symmetric about the mean.
"""
import numpy as np

def mean(data):
    """

    :param data: 

    """
    return np.mean(data)

def variance(data):
    """

    :param data: 

    """
    return np.var(data)

def stddev(data):
    """

    :param data: 

    """
    return np.std(data)

def normal_samples(mu=0, sigma=1, n=5):
    """

    :param mu: Default value = 0)
    :param sigma: Default value = 1)
    :param n: Default value = 5)

    """
    return np.random.normal(loc=mu, scale=sigma, size=n)

# Example usage
if __name__ == "__main__":
    nums = [1, 2, 3, 4, 5]
    print("Mean:", mean(nums))
    print("Variance:", variance(nums))
    print("Std Dev:", stddev(nums))
    print("Normal samples:", normal_samples())
