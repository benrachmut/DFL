from config_ import *


def all_compute(clients_dict,t):
    for client in clients_dict.values():
        client.compute(t)


def all_send(clients_dict):
    for id_, client in clients_dict.items():
        sender = id_
        what_to_send = client.data_to_send
        receivers = client.neighbors

        for receiver_id in receivers:
            receiver_obj = clients_dict[receiver_id]
            receiver_obj.received_data[sender] = what_to_send


def all_receive(clients_dict):
    for client in clients_dict.values():
        client.digest_received_data()

def all_init(clients_dict):
    for client in clients_dict.values():
        client.initialize()

def measure_neighbors(clients_dict,t):
    for id_,client in clients_dict.items():
        neighbors_ids = client.neighbors
        client_test_data = client.test_data


def execute_distributed(clients_dict):
    all_init(clients_dict)
    measure_neighbors(0)
    for t in range(1,ec.iterations):
        print("*********** iteration: ",t," ***********")
        all_send(clients_dict)
        all_receive(clients_dict)
        all_compute(clients_dict,t)













def execute(clients):
    if ec.environment == Env.Distributed:
        execute_distributed(clients)
