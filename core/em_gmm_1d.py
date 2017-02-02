import numpy as np
from cluster import Cluster
import math


def cluster(data, n_cluster):
    """
    data should be in numpy array of matrix(x, y)
    ex if there is 2 data [0, 5] and [8, 2]
    the matrix should be [[0, 5], [8, 2]]
    """

    # initialization
    min_data = np.min(data, axis=0)
    max_data = np.max(data, axis=0)
    clusters = initialize_cluster(n_cluster, features_num, min_data, max_data)

def initialize_cluster(n_cluster, features_num, min_data, max_data):
    clusters = []
    for i in range(n_cluster):
        cluster_i = Cluster(features_num)
        clusters.append(cluster_i.initialize(min_data, max_data))
    return clusters

def update_cluster(data, clusters):
    """
    data should be in numpy array of matrix
    clusters should be a list of Cluster
    notice that the code is only works for 1d
    """

    # finding P(x | n)
    pxn = []
    for datum in data:
        pxi = []
        datum_val = datum[0]
        for cluster in clusters:
            mean = cluster.mean[0]
            variance = cluster.covariance[0][0]
            pxic = 1. / (math.sqrt(2. * math.pi * variance))
            pxic *= math.exp((- (datum_val - mean) ** 2.)/(2. * variance))
            pxi.append(pxic)
        pxn.append(pxi)

    # finding P(n | x)
    # first, find the lower part
    joint = []
    for pxi in pxn:
        joint_element = 0.
        for i in range(len(clusters)):
            joint_element += pxi[i] * clusters[i].prior[0]
        joint.append(joint_element)

    pnx = []
    for i in range(len(clusters)):
        cluster = clusters[i]
        pix = []
        for pxi in pxn:
            pixc = pxi[i] * cluster.prior[0]
            pix.append(pixc)
        pnx.append(pix)

    # update the mean
    for i in range(len(clusters)):
        cluster = clusters[i]
        upper = 0.
        lower = 0.
        pix = pnx[i]
        for j in range(len(data)):
            datum_val = datum[j][0]
            pixc = pix[j]
            upper += pixc * datum_val
            lower += pixc
        if lower > 0 or lower < 0:
            cluster.mean[0] = upper / lower

    # update the variance
    for i in range(len(clusters)):
        cluster = clusters[i]
        upper = 0.
        lower = 0.
        pix = pnx[i]
        for j in range(len(data)):
            datum_val = datum[j][0]
            pixc = pix[j]
            upper += pixc * ((datum_val - cluster.mean[0])**2)
            lower += pixc
        if lower > 0 or lower < 0:
            cluster.covariance[0][0] = upper / lower

    # update the prior
    for i in range(len(clusters)):
        cluster = clusters[i]
        pix = pnx[i]
        prior = 0.
        for pixc in pix:
            prior += pixc
        prior /= len(data)
        cluster.prior[0] = prior

