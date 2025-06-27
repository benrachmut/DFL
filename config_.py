from enum import Enum

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ExperimentConfig:
    def __init__(self):
        self.data_set = None
        self.topology_technique = None
        self.num_clients = None
        self.num_of_hubs = None
        self.num_runs = None
        self.num_run = None
        self.data_distribution = None
        self.unlabeled_data_percentage = None
        self.algorithm = None
        self.environment = None
        self.iterations = None
        self.client_non_hub_net = None
        self.client_hub_net = None
        self.num_classes =None

    def update_vgg(self):
        self.batch_size = 64
        self.learning_rate_fine_tune = 0.0001
        self.learning_rate_train = 0.0001
        self.epochs_num = 5

    def update_alexNet(self):
        self.batch_size = 64
        self.learning_rate_fine_tune = 0.001
        self.learning_rate_train = 0.0001
        self.epochs_num = 5

ec = ExperimentConfig()

class DataSet(Enum):
    CIFAR100 = "CIFAR100"
    CIFAR10 = "CIFAR10"

class DataDistribution(Enum):
    IID = "IID"

class TopologyTechnique(Enum):
    DenseRandom = "DenseRandom"
    SparseRandom = "SparseRandom"

class Algorithm(Enum):
    DMAPL ="DMAPL"

class Env(Enum):
    Distributed ="Distributed"

class NetType(Enum):
    AlexNet = "AlexNet"
    VGG = "VGG"