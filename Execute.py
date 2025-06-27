from config_ import *


def execute_distributed(clients_dict):
    for client in clients_dict.values():
        client.initialize()

    for i in ec.iterations:
        data_to_send = {id_: {}}  # what to send, list to who
        for client in clients_dict.values():
            id_ = client.id_
            data_to_send[id_][client.data_to_send] = client.neighbors

        for client in clients_dict.values():
            sender = client.id_
            for what_to_send, receivers in data_to_send[sender]:
                for receiver_id in receivers:
                    client_to_receive = clients_dict[receiver_id]
                    client_to_receive.recieve_data(what_to_send)

        for client in clients_dict.values():
            client.digest_recieved_data()


        for client in clients_dict.values():
            client.compute()




def execute(clients):
    if ec.environment == Env.Distributed:
        execute_distributed(clients)
