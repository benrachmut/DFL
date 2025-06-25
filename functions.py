from random import Random

from sympy.core.parameters import distribute

from config_ import *

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split

import math
from collections import deque

def create_random_uniform_dict(p):
    num_clients = ec.num_clients
    rng = Random(ec.num_run)  # create independent random instance

    neighbors_dict = {i: [] for i in range(num_clients)}

    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            if rng.random() < p:
                neighbors_dict[i].append(j)
                neighbors_dict[j].append(i)

    # Step 2: Fix isolated clients
    for i in range(num_clients):
        if len(neighbors_dict[i]) == 0:
            # Pick a random other client
            j = rng.choice([x for x in range(num_clients) if x != i])
            neighbors_dict[i].append(j)
            neighbors_dict[j].append(i)

    return neighbors_dict


def get_neighbors_dict():
    print(ec.topology_technique)
    if ec.topology_technique == TopologyTechnique.DenseRandom:
        return create_random_uniform_dict(0.7)
    if ec.topology_technique == TopologyTechnique.SparseRandom:
        return create_random_uniform_dict(0.2)




def compute_distances(graph, start):
    """Breadth-first search to compute shortest path lengths from start to all other nodes."""
    distances = {node: math.inf for node in graph}
    distances[start] = 0
    queue = deque([start])

    while queue:
        current = queue.popleft()
        for neighbor in graph[current]:
            if distances[neighbor] == math.inf:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)

    return distances

def select_hubs():
    rng = Random(ec.num_run)
    all_clients = list( ec.neighbors_dict.keys())
    selected = []

    while len(selected) < ec.num_of_hubs:
        scores = []

        for agent in all_clients:
            if agent in selected:
                continue

            # Degree factor (more neighbors = higher score)
            degree_score = len( ec.neighbors_dict[agent]) + 1  # avoid zero

            # Distance factor: penalize proximity to already selected agents
            if selected:
                min_dist = math.inf
                for sel in selected:
                    distances = compute_distances( ec.neighbors_dict, sel)
                    if distances[agent] < min_dist:
                        min_dist = distances[agent]
                dist_score = min_dist if min_dist < math.inf else 1000
            else:
                dist_score = 1  # no penalty for first pick

            # Final score: higher = more likely to be picked
            # inverse distance makes close nodes score lower
            final_score = degree_score / dist_score
            scores.append((agent, final_score))

        # Normalize scores to probabilities
        total_score = sum(score for _, score in scores)
        probabilities = [(agent, score / total_score) for agent, score in scores]

        # Roulette wheel selection
        r = rng.random()
        cumulative = 0
        for agent, prob in probabilities:
            cumulative += prob
            if r < cumulative:
                selected.append(agent)
                break

    return selected

# Press the green button in the gutter to run the script.



def create_data():
    torch.manual_seed(ec.num_run)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # CIFAR-10 normalization
    ])

    # Download CIFAR-10 train set
    if ec.data_set == DataSet.CIFAR10:
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


    ul_data, client_pool = split_train_label_unlabel(train_dataset)


    if ec.data_distribution == DataDistribution.IID:
        client_data = random_split(
            client_pool,
            [ec.num_clients // ec.num_clients] * ec.num_clients,
            generator=torch.Generator().manual_seed(ec.num_run + 1)
        )
    print()

def split_train_label_unlabel(train_dataset):
    total_size = len(train_dataset)
    global_size = int(ec.unlabeled_data_percentage * total_size)
    client_size = total_size - global_size

    global_data, client_pool = random_split(train_dataset, [global_size, client_size],
                                            generator=torch.Generator().manual_seed(ec.num_run))

    return global_data, client_pool

