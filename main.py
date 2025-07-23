from functions import *
import pickle


def extract_info(clients,t):
    measure_neighbors(clients, t)

    rd = RecordData(clients)
    data_to_pickle[ec.data_set.name][ec.topology_technique][ec.num_clients][ec.client_hub_net] = rd
    name_pkl = str(ec)+ ".pkl"
    with open(name_pkl, 'wb') as f:
        pickle.dump(data_to_pickle, f)


if __name__ == '__main__':
    print("Device:", device)

    data_sets = [DataSet.CIFAR100]
    topologies = [TopologyTechnique.DenseRandom]
    client_sizes = [25]
    client_hub_nets = [NetType.VGG,NetType.AlexNet]

    ec.num_of_hubs = 5
    ec.num_runs = 1
    ec.data_distribution = DataDistribution.IID
    ec.unlabeled_data_percentage = 0.2
    ec.iterations = 10
    ec.client_non_hub_net = NetType.AlexNet

    ec.environment = Env.Distributed
    ec.algorithm = Algorithm.DMAPL
    data_to_pickle = {}


    for data_set in data_sets:
        ec.data_set = data_set
        data_to_pickle[ec.data_set.name] = {}

        for topo in topologies:
            ec.topology_technique =topo
            data_to_pickle[ec.data_set.name][ec.topology_technique] = {}

            for client_size  in client_sizes:
                ec.num_clients = client_size
                data_to_pickle[ec.data_set.name][ec.topology_technique][ec.num_clients] = {}

                for client_hub_net in client_hub_nets:
                    ec.client_hub_net = client_hub_net
                    data_to_pickle[ec.data_set.name][ec.topology_technique][ec.num_clients][ec.client_hub_net] = {}



                    for num_run in range(ec.num_runs):
                        #data_to_pickle[ec.algorithm][ec.client_hub_net][num_run] = {}

                        ec.num_run = num_run
                        ec.neighbors_dict = get_neighbors_dict()
                        ec.selected_hubs = select_hubs()
                        data_per_client_dict = create_data()
                        clients = create_clients(data_per_client_dict)

                        if ec.environment == Env.Distributed:
                            all_init(clients)
                            extract_info(clients,0)

                            for t in range(1, ec.iterations):
                                print("*********** iteration:",t,"***********")
                                all_send(clients)
                                all_receive(clients)
                                all_compute(clients, t)
                                extract_info(clients,t)





        #execute(clients)
