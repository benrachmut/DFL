# Replace 'filename.pkl' with your actual file name
import pickle

from config_ import Algorithm, NetType


def get_avg_per_client(param):
    nums = getattr(selected_data, param)
    dict_all_iters = {}
    for client_id,dict_per_iter in nums.items():
        for i,result_ in dict_per_iter.items():
            if i not in dict_all_iters:
                dict_all_iters[i]=[]
            dict_all_iters[i].append(result_)
    ans = {}
    for i, num_list in dict_all_iters.items():
        ans[i] = sum(num_list)/len(num_list)
    return ans


if __name__ == '__main__':

    with open('CIFAR10,SparseRandom,25,5,1,10,IID,DMAPL,Distributed,VGG,AlexNet.pkl', 'rb') as f:
        data = pickle.load(f)

    print(data)

    algos =  [Algorithm.DMAPL]
    strong_nets = [NetType.VGG]
    for algo in algos:
        for strong_net in strong_nets:
            selected_data = data[algo][strong_net][0]
            avg_best_n = get_avg_per_client("client_best_neighbor_model_accuracy_1")
            avg_personal = get_avg_per_client("client_self_model_accuracy_1")
            #avg_best_n = get_avg_per_client("client_loss_test")
            #avg_personal = get_avg_per_client("client_loss_train")
            print()