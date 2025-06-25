
from config_ import *
from functions import *



if __name__ == '__main__':
    print("Device:", device)
    ec.data_set = DataSet.CIFAR10
    ec.topology_technique = TopologyTechnique.SparseRandom
    ec.num_clients = 25
    ec.num_of_hubs = 5
    ec.num_runs = 1
    ec.data_distribution = DataDistribution.IID
    ec.unlabeled_data_percentage = 0.2

    for num_run in range(ec.num_runs):
        ec.num_run = num_run
        ec.neighbors_dict = get_neighbors_dict()
        ec.selected_hubs = select_hubs()

        train_data,test_data,ul_data = create_data()
        #clients = create_clients()

