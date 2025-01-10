import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
import community.community_louvain as community_louvain
import networkx as nx
import os
from .assortnet import *

def cal_disentanglement_dis(latent_reps: np.ndarray, concept_labels: np.ndarray):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    latent_reps = torch.from_numpy(latent_reps)
    latent_reps.to(device)
    concept_labels = torch.from_numpy(concept_labels)
    concept_labels.to(device)

    # Distance between same concept
    dis_same_pos = torch.cdist(latent_reps[concept_labels == 1], latent_reps[concept_labels == 1])
    # dis_same_neg = torch.cdist(latent_reps[concept_labels == 0], latent_reps[concept_labels == 0])
    # dis_same = (dis_same_pos.sum() + dis_same_neg.sum()) / (dis_same_pos.shape[0] + dis_same_neg.shape[0])
    dis_same = dis_same_pos.sum(axis=1) / dis_same_pos.shape[1]
    dis_same = dis_same.mean()

    # Distance between different concept
    dis_diff = torch.cdist(latent_reps[concept_labels == 1], latent_reps[concept_labels == 0])
    # dis_diff = dis_diff.sum() / dis_diff.shape[0]
    dis_diff = dis_diff.sum(axis=1) / dis_diff.shape[1]
    dis_diff = dis_diff.mean()

    D = (dis_diff - dis_same) / (dis_same + dis_diff)

    return float(D)


def cal_disentanglement_svm(latent_reps: np.ndarray, concept_labels: np.ndarray):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    concept_presented_points = latent_reps[concept_labels == 1]
    concept_absent_points = latent_reps[concept_labels == 0]
    k = max(int(len(concept_presented_points) * 0.4), 5)

    # Find k nearest neighbors and construct a graph
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(concept_presented_points)
    dis, indices = knn.kneighbors(concept_presented_points, return_distance=True)
    
    G = nx.Graph()
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors:
            G.add_edge(i, neighbor)

    # Perform Graph Community Detection
    partition = community_louvain.best_partition(G)
    clusters = {}
    for node, cluster_id in partition.items():
        if cluster_id not in clusters:
            clusters[cluster_id] = torch.tensor(np.array([concept_presented_points[node]])).to(device)
        else:
            clusters[cluster_id] = torch.cat([clusters[cluster_id], torch.tensor(np.array([concept_presented_points[node]])).to(device)], dim=0)

    # average distance between points in the same cluster
    avg_dis = torch.zeros(len(clusters))
    for cluster_id, cluster in clusters.items():
        dis = torch.cdist(cluster, cluster)
        avg_dis[cluster_id] = (dis.sum(axis=1) / dis.shape[1]).mean()

    # Calculate the disentanglement
    num_absent_points_inside = 0
    concept_absent_points = torch.from_numpy(concept_absent_points).to(device)
    for absent_point in concept_absent_points:
        for cluster_id, cluster in clusters.items():
            dis = torch.cdist(cluster, absent_point.unsqueeze(0))
            if (dis.min() <= avg_dis[cluster_id]):
                num_absent_points_inside += 1
                break
    
    D = 1 - num_absent_points_inside / concept_absent_points.shape[0]
    return D

def cal_disentanglement_assort(latent_reps: np.ndarray, concept_labels: np.ndarray):
    concept_presented_points = latent_reps[concept_labels == 1]
    concept_absent_points = latent_reps[concept_labels == 0]
    k = max(int(len(concept_presented_points) * 0.9), 5)

    G = nx.Graph()
    i = 0
    for _ in range(len(concept_presented_points)):
        G.add_node(i, concept='present')
        i += 1
    for _ in range(len(concept_absent_points)):
        G.add_node(i, concept='absent')
        i += 1
    
    points = np.concatenate((concept_presented_points, concept_absent_points), axis=0)
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(points)
    dis, indices = knn.kneighbors(points, return_distance=True)
    dis = np.abs(dis)
    max_dis = np.array(dis).max()
    min_dis = np.array(dis).min()
    
    added = set()
    for i, (neighbors, d) in enumerate(zip(indices, dis)):
        for neighbor, dd in zip(neighbors, d):
            a = min(i, neighbor)
            b = max(i, neighbor)
            if dd == 0 or (a, b) in added:
                continue
            weight = (1 - ((dd - min_dis) / (max_dis - min_dis)))
            # weight = 1 / dd
            G.add_weighted_edges_from([(i, neighbor, weight)])
            added.add((a, b))
        
    return assortment_discrete(G, "concept", ["present", "absent"])

def cal_disentanglement_avgdis(latent_reps: np.ndarray, concept_labels: np.ndarray,):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    latent_reps = torch.from_numpy(latent_reps)
    latent_reps.to(device)
    concept_labels = torch.from_numpy(concept_labels)
    concept_labels.to(device)
    # All distances
    dis = torch.cdist(latent_reps, latent_reps)

    # Find the maximum distance
    max_dis = dis.max()

    # Normalize the distance
    dis /= max_dis
    
    # Distance between same concept
    disentanglement = torch.zeros(concept_labels.shape[0])

    for i in range(concept_labels.shape[0]):
        label = concept_labels[i]
        dis_same = dis[i][concept_labels == label]
        dis_diff = dis[i][concept_labels != label]
        disentanglement[i] = dis_diff.min() - dis_same.min()

    return float(disentanglement.mean())