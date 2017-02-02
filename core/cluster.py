import numpy as np


class Cluster(object):
    """
    An object representing cluster for EM algorithm, with attributes are array based on features of data
    Attributes:
        prior: the probability of a cluster
        mean: the mean of a cluster
        covariance: the covariance of a cluster

    TODO: error handling for wacky arguments
    """

    def __init__(self, features_num, prior=None, mean=None, covariance=None):
        self.prior = prior if prior is not None else np.empty([features_num])
        self.mean = mean if mean is not None else np.empty([features_num])
        self.covariance = covariance if covariance is not None else np.empty([features_num, features_num])
        self.features_num = features_num if features_num > 0 else 1

    def initialize(self, min_val, max_val):
        self.prior = np.ones([self.features_num]) * (1./self.features_num)

        # get range of data
        range_val = max_val - min_val

        self.mean = (np.random.rand(self.features_num) * range_val) + min_val
        self.covariance = np.random.rand(self.features_num, self.features_num)

    def update_mean():
        self.mean = []

    def update_prior():
        self.prior = []

    def update_covariance():
        self.covariance = []
