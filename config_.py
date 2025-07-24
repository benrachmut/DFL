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

        self.hub_ratio = None
        self.epochs_num = 5
    def __str__(self):
        a=self.data_set.name
        b=self.topology_technique.name
        c=str(self.num_clients)
        d=str(int(self.hub_ratio*100))
        e=str(self.num_runs)
        f=str(self.iterations)
        g=self.data_distribution.name
        h=self.algorithm.name
        i=self.environment.name
        x=self.client_hub_net.name
        y=self.client_non_hub_net.name
        return a+","+b+","+c+","+d+","+e+","+f+","+g+","+h+","+i+","+x+","+y

    def update_vgg(self):
        batch_size = 128
        learning_rate= 0.0001
        return batch_size,learning_rate

    def update_alexNet(self):
        batch_size = 64
        learning_rate = 0.001
        return batch_size, learning_rate

    def to_dict(self):
        """Returns a dictionary of attribute names and their values."""
        return {attr: getattr(self, attr) for attr in dir(self) if
                not callable(getattr(self, attr)) and not attr.startswith("__")}


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