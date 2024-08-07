# RBF Neural Network Implementation

## Overview

This code implements a Radial Basis Function (RBF) Neural Network for function approximation. Specifically, it:

1. Implements core functionality of an RBF Neural Network
2. Uses K-means algorithm to select RBF centers
3. Trains the network to fit given data points
4. Uses the trained network for predictions
5. Visualizes the fitting results

## Theoretical Background

An RBF Neural Network is a type of feedforward neural network consisting of three layers:

1. Input Layer: Receives input data
2. Hidden Layer: Uses radial basis functions (typically Gaussian) as activation functions
3. Output Layer: Computes a linear combination of hidden layer outputs

The RBF network operates by:

1. Calculating distances between inputs and RBF centers
2. Transforming these distances into activation values using Gaussian functions
3. Multiplying activations by weights and summing to produce final output

Advantages:
- Fast training
- Good approximation of arbitrary non-linear functions
- Less susceptible to local minima

## Implementation Process

1. Define RBFN class with initialization, Gaussian calculation, training, and prediction methods
2. Generate test data (quadratic function with random noise)
3. Create RBF network instance and train
4. Use trained network for predictions
5. Plot original data and prediction results for comparison

## Core Code Explanation

### RBFN Class

```python
class RBFN:
    def __init__(self, in_dim, hid_dim, out_dim):
        # Initialize network structure and parameters
        ...

    def _calc_Gaussian(self, X):
        # Calculate Gaussian activation values
        ...

    def train(self, X, Y):
        # Train the network
        # 1. Use K-means to select centers
        # 2. Calculate beta parameter
        # 3. Compute weights
        ...

    def predict(self, X):
        # Use trained network for predictions
        ...
```

### Key Steps

1. Center Selection:
   ```python
   k_means = KMeans(init='k-means++', n_clusters=self.hid_dim, n_init=10)
   k_means.fit(np.array(X))
   self.centers = np.array(k_means.cluster_centers_)
   ```

2. Beta Calculation:
   ```python
   c_max = np.max([np.linalg.norm(u - v) for u in self.centers for v in self.centers])
   self.beta = self.hid_dim / (c_max ** 2)
   ```

3. Weight Calculation:
   ```python
   self.W = np.dot(np.linalg.pinv(Gaussian), Y)
   ```

## Results

Running the code generates an image showing original data points and the RBF network's fitting results.



Blue asterisks represent original data points, while red dots show the RBF network's predictions. Note how the RBF network effectively fits the overall trend despite noise in the original data.

## Usage Instructions

1. Ensure necessary libraries are installed: sklearn, matplotlib, numpy
2. Copy entire code into a Python environment and run
3. Adjust network performance by modifying:
   - `hid_dim`: Number of hidden layer nodes (RBF centers)
   - Test data generation method and quantity

## Notes

- This implementation is primarily for educational and demonstration purposes. Further optimization may be needed for large-scale or complex problems.
- RBF network performance largely depends on center selection. Experiment with other center selection methods to improve performance.
- For high-dimensional inputs, increasing the number of hidden layer nodes may be necessary for better fitting.

## References

- Bishop, C. M. (1995). Neural Networks for Pattern Recognition. Oxford University Press.
- Haykin, S. (1998). Neural Networks: A Comprehensive Foundation. Prentice Hall.
