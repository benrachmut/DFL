from enum import Enum

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ExperimentConfig:
    def __init__(self):
        data_set = None
        topology_technique = None
        num_clients = None
        num_of_hubs = None
        num_runs = None
        num_run = None
        data_distribution = None
        unlabeled_data_percentage = None

ec = ExperimentConfig()

class DataSet(Enum):
    CIFAR100 = "CIFAR100"
    CIFAR10 = "CIFAR10"

class DataDistribution(Enum):
    IID = "IID"

class TopologyTechnique(Enum):
    DenseRandom = "DenseRandom"
    SparseRandom = "SparseRandom"



