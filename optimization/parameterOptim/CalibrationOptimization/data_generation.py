import numpy as np
import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import random
from scipy.interpolate import griddata

class DataGeneration:
    def __init__(self, data, probabilities,number_of_new_points = None):
        self.data = np.array(data)
        self.probabilities = np.array(probabilities)
        self._prepare_data()
        self._calculate_wcss()
        self.number_of_optimal_clusters = self._find_optimal_clusters()
        if number_of_new_points == None:
            self.number_of_new_points = len(data)
        else:
            self.number_of_new_points = number_of_new_points
        self.new_points = self._generate_new_points()
        self.new_probabilities = self._estimate_probabilities()
        self.new_pdf = {tuple(self.new_points[i]): self.new_probabilities[i] for i in range(len(self.new_probabilities))}

    def _prepare_data(self):
        if self.data.ndim == 1:
            self.data = self.data.reshape(-1, 1)

    def _calculate_wcss(self):
        self.wcss = []
        max_clusters = min(len(self.data), 10)
        for i in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, max_iter=300, n_init=10, random_state=0)
            kmeans.fit(self.data)
            self.wcss.append(kmeans.inertia_)

    def _find_optimal_clusters(self):
        line = np.linspace(self.wcss[0], self.wcss[-1], len(self.wcss))
        distances = np.abs(self.wcss - line)
        return np.argmax(distances) + 1

    def _generate_new_points(self):
        kmeans = KMeans(n_clusters=self.number_of_optimal_clusters, random_state=0).fit(self.data)
        labels = kmeans.labels_
        cluster_weights = [0] * self.number_of_optimal_clusters
        for label, prob in zip(labels, self.probabilities):
            cluster_weights[label] += prob

        new_points = []
        for _ in range(self.number_of_new_points):
            cluster_index = random.choices(range(self.number_of_optimal_clusters), weights=cluster_weights, k=1)[0]
            cluster_points = self.data[labels == cluster_index]
            min_vals = np.min(cluster_points, axis=0)
            max_vals = np.max(cluster_points, axis=0)
            new_point = [random.uniform(min_val, max_val) for min_val, max_val in zip(min_vals, max_vals)]
            new_points.append(new_point)
        return np.array(new_points)

    
    # def _generate_new_points(self,exploration_factor=0.1):
    #     """
    #     Generate new points with a balance between following the original probabilities and exploring the space.

    #     :param exploration_factor: Determines the degree of random exploration in point generation.
    #     """
    #     kmeans = KMeans(n_clusters=self.number_of_optimal_clusters, random_state=0).fit(self.data)
    #     labels = kmeans.labels_

    #     # Adjusting cluster weights
    #     adjusted_weights = [np.sqrt(prob) for prob in self.probabilities]
    #     total_weight = sum(adjusted_weights)
    #     cluster_weights = [0] * self.number_of_optimal_clusters
    #     for label, weight in zip(labels, adjusted_weights):
    #         cluster_weights[label] += weight/total_weight

    #     new_points = []
    #     for _ in range(self.number_of_new_points):
    #         if random.random() < exploration_factor:
    #             # Random exploration
    #             cluster_index = random.choice(range(self.number_of_optimal_clusters))
    #         else:
    #             # Weighted selection
    #             cluster_index = random.choices(range(self.number_of_optimal_clusters), weights=cluster_weights, k=1)[0]

    #         cluster_points = self.data[labels == cluster_index]
    #         min_vals = np.min(cluster_points, axis=0)
    #         max_vals = np.max(cluster_points, axis=0)
    #         new_point = [random.uniform(min_val, max_val) for min_val, max_val in zip(min_vals, max_vals)]
    #         new_points.append(new_point)

    #     return np.array(new_points)





    def _estimate_probabilities(self):
        #GAUSSIAN ESTIMATE
        #################
        
        # nbrs = NearestNeighbors(n_neighbors=2).fit(self.data)
        # estimated_probs = []
        # for point in self.new_points:
        #     distances, indices = nbrs.kneighbors([point])
        #     point_a_index = indices[0, 0]
        #     point_b_index = nbrs.kneighbors([self.data[point_a_index]], return_distance=False)[0, 1]
        #     std_dev = np.linalg.norm(self.data[point_a_index] - self.data[point_b_index]) / 2
        #     prob = self.probabilities[point_a_index] * stats.norm.pdf(distances[0, 0], 0, std_dev)
        #     estimated_probs.append(prob)
        

        #CLOSEST ESTIMATE
        #######################
        # nbrs = NearestNeighbors(n_neighbors=2).fit(self.data)
        # estimated_probs = []

        # for point in self.new_points:
        #     distances, indices = nbrs.kneighbors([point])
        #     prob = np.mean([self.probabilities[idx] for idx in indices[0]])
        #     estimated_probs.append(prob)
        # return np.array(estimated_probs) / sum(estimated_probs)
        #INTERPOLATE ESTIMATE
        ###########################
        estimated_probs = griddata(self.data, self.probabilities, self.new_points, method='linear')
        # Handling any NaN values that may arise if new points fall outside the convex hull of original points
        estimated_probs = np.nan_to_num(estimated_probs, nan=np.min(self.probabilities))

        # Normalizing probabilities
        estimated_probs /= np.sum(estimated_probs)
        return estimated_probs

    @property
    def data_generated(self):
        return self.new_points

    @property
    def probabilities_generated(self):
        return self.new_probabilities

    @property
    def pdf_generated(self):
        return self.new_pdf
    
