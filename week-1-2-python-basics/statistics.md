# Statistics Notes

## Mean
The average value: mean = (sum of all values) / (number of values)

## Variance
A measure of how spread out numbers are.

## Probability
Probability is the measure of the likelihood of an event.

## Normal Distribution
A bell-shaped curve, symmetric about the mean.

---

## Example (Python)
```python
import numpy as np
nums = [1, 2, 3, 4, 5]
print('Mean:', np.mean(nums))
print('Variance:', np.var(nums))
# Normal distribution
samples = np.random.normal(loc=0, scale=1, size=5)
print('Normal distribution samples:', samples)
```
