import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class DataGeneration:
    def __init__(self, data_points, densities, number_of_new_points, bounds):
        """
        data_points: n-dimensional points
        densities: corresponding discrete probabilities
        bounds: tuple of (min, max) for each dimension
        """
        self.data_points = np.array(data_points)
        self.densities = np.array(densities)
        self.bounds = np.array(bounds)
        self.kde = gaussian_kde(self.data_points.T, weights=self.densities)
        self.number_of_new_points = number_of_new_points

    def _is_dense(self, threshold=0.75):
        """
        Checks if the current points are dense based on a threshold.
        """
        return np.var(self.densities) > threshold

    def _within_bounds(self, points):
        """
        Filter points to keep only those within the specified bounds.
        """
        within_bounds = np.all((points >= self.bounds[:, 0]) & (points <= self.bounds[:, 1]), axis=1)
        return points[within_bounds]

    def data_generated(self, exploration_factor=0.25):
        """
        Generate new points based on the continuous probability distribution.
        Explore new space if points are dense.
        """
        num_points = self.number_of_new_points
        if self._is_dense():
            exploration_points = int(num_points * exploration_factor)
            normal_points = num_points - exploration_points
        else:
            normal_points = num_points
            exploration_points = 0

        # Generating points from the existing distribution
        new_points = self.kde.resample(normal_points)

        # Exploring new space
        if exploration_points:
            cov_factor = np.cov(self.data_points.T) * 1.5
            explore_kde = gaussian_kde(self.data_points.T, bw_method=cov_factor)
            explore_points = explore_kde.resample(exploration_points)
            new_points = np.hstack((new_points, explore_points))

        # Ensuring points are within global bounds
        self.new_points = self._within_bounds(new_points.T)

        return self.new_points

    def probabilities_generated(self, points):
        """
        Calculate the probability density of each point using KDE.
        """
        densities = self.kde(points.T)
        # Normalize to make it sum to 1, treating densities as relative probabilities
        self.normalized_densities = densities / np.sum(densities)
        return self.normalized_densities
    
    # @property
    # def data_generated(self):
    #     return self.new_points

    # @property
    # def probabilities_generated(self):
    #     return self.normalized_densities