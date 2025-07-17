from random import Random
from Entities import *
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


def distributed_iid(data_pool):
    total_len = len(data_pool)
    base_len = total_len // ec.num_clients
    lengths = [base_len] * ec.num_clients
    lengths[-1] += total_len - sum(lengths)  # Assign remainder to last client

    client_data = random_split(
        data_pool,
        lengths,
        generator=torch.Generator().manual_seed(ec.num_run + 1)
    )
    return client_data

def get_data_sets():
    if ec.data_set == DataSet.CIFAR10:
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)  # Adjust for CIFAR-10
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        ec.num_classes = 10

    if ec.data_set == DataSet.CIFAR100:
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)  # Adjust for CIFAR-10
        ])
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        ec.num_classes = 100

    return train_dataset,test_dataset

def distribute_data_between_clients(train_dataset,test_dataset):
    if ec.data_distribution == DataDistribution.IID:
        distributed_client_data = distributed_iid(train_dataset)
        distributed_test_dataset = distributed_iid(test_dataset)
    return   distributed_client_data, distributed_test_dataset


def create_data_per_client_dict(train_data, test_data, ul_data,test_dataset):
    ans = {}
    for i in range(ec.num_clients):
        ans[i] = {"train_data": train_data[i], "test_data": test_data[i], "UL": ul_data,"global_test":test_dataset}

    return ans

def create_data():
    # Set seed for reproducibility
    torch.manual_seed(ec.num_run)

    # Define transform

    # Load CIFAR-10 dataset
    train_dataset,test_dataset = get_data_sets()
    global_data_set =Subset(test_dataset, list(range(len(test_dataset))))
    # Split into labeled/unlabeled pools
    ul_data, client_pool = split_train_label_unlabel(train_dataset)

    distributed_client_data, distributed_test_dataset =distribute_data_between_clients(client_pool,test_dataset)
    # Split IID data among clients

    dict_data_per_client = create_data_per_client_dict(distributed_client_data, distributed_test_dataset, ul_data,global_data_set)
    return   dict_data_per_client

def create_clients(data_per_client_dict):
    # Extract known keys
    if ec.algorithm ==  Algorithm.DMAPL:
        clients_list = [
            Client_DMAPLE(customer_id, dataset_dict)
            for customer_id, dataset_dict in data_per_client_dict.items()
        ]
        ans =  {}
        for c in clients_list:
            ans[c.id_]=c
    return ans



def split_train_label_unlabel(train_dataset):
    total_size = len(train_dataset)
    global_size = int(ec.unlabeled_data_percentage * total_size)
    client_size = total_size - global_size

    global_data, client_pool = random_split(train_dataset, [global_size, client_size],
                                            generator=torch.Generator().manual_seed(ec.num_run))

    return global_data, client_pool

