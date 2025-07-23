# Replace 'filename.pkl' with your actual file name
import pickle

from config_ import Algorithm, NetType


def get_avg_per_client(selected_data,param):
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


def create_data_per_measure(measure_ ):
    with open('CIFAR10,SparseRandom,25,5,1,10,IID,DMAPL,Distributed,VGG,AlexNet.pkl', 'rb') as f:
        data = pickle.load(f)
    for algo in algos:
        for strong_net in [NetType.VGG]:
            selected_data = data[algo][strong_net][0]
            if "With Hubs" not in for_graph[25]:
                for_graph[25]["With Hubs"] = {}
            for_graph[25]["With Hubs"][alt_measure_names[measure_]] = get_avg_per_client(selected_data,measure_)

    with open('CIFAR10,SparseRandom,25,5,1,10,IID,DMAPL,Distributed,AlexNet,AlexNet.pkl', 'rb') as f:
        data = pickle.load(f)
    for algo in algos:
        for strong_net in [NetType.AlexNet]:
            selected_data = data[algo][strong_net][0]
            if "Without Hubs" not in for_graph[25]:
                for_graph[25]["Without Hubs"] = {}

            for_graph[25]["Without Hubs"][alt_measure_names[measure_]] = get_avg_per_client(selected_data,measure_)

    with open('CIFAR10,SparseRandom,10,5,1,10,IID,DMAPL,Distributed,VGG,AlexNet.pkl', 'rb') as f:
        data = pickle.load(f)
    for algo in algos:
        for strong_net in [NetType.VGG]:
            selected_data = data[algo][strong_net][0]
            if "With Hubs" not in for_graph[10]:
                for_graph[10]["With Hubs"] = {}
            for_graph[10]["With Hubs"][alt_measure_names[measure_]] = get_avg_per_client(selected_data,measure_)

    with open('CIFAR10,SparseRandom,10,5,1,10,IID,DMAPL,Distributed,AlexNet,AlexNet.pkl', 'rb') as f:
        data = pickle.load(f)
    for algo in algos:
        for strong_net in [NetType.AlexNet]:
            selected_data = data[algo][strong_net][0]
            if "Without Hubs" not in for_graph[10]:
                for_graph[10]["Without Hubs"] = {}

            for_graph[10]["Without Hubs"][alt_measure_names[measure_]] = get_avg_per_client(selected_data,measure_)

import matplotlib.pyplot as plt

def plot_nested_dict(data, x_label="X-axis", y_label="Y-axis", title="Graph"):
    """
    Plots a nested dictionary where:
      - The outer key (e.g., 'with hub', 'without hub') determines the color.
      - The middle key ('self' or 'Best Neighbor') determines the line style.
      - The innermost dictionary maps x-values to y-values.

    Args:
        data (dict): Nested dictionary to plot.
        x_label (str): Label for the X-axis.
        y_label (str): Label for the Y-axis.
        title (str): Title of the plot.
    """

    # Assign unique colors to outer keys
    color_palette = plt.get_cmap("tab10")
    outer_keys = list(data.keys())
    colors = {key: color_palette(i) for i, key in enumerate(outer_keys)}

    # Define line styles
    linestyles = {
        'self': 'solid',
        'Best Neighbor': 'dashed'
    }

    plt.figure(figsize=(8, 5))

    for outer_key, inner_dict in data.items():
        for line_type, xy_vals in inner_dict.items():
            x = list(xy_vals.keys())
            y = list(xy_vals.values())
            plt.plot(
                x, y,
                label=f"{outer_key} - {line_type}",
                color=colors[outer_key],
                linestyle=linestyles.get(line_type, 'solid')  # Default to solid
            )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':


    algos =  [Algorithm.DMAPL]
    strong_nets = [NetType.VGG]

    for_graph = {}
    for_graph[25] = {}
    for_graph[10] = {}
    alt_measure_names = {"client_best_neighbor_model_accuracy_1":"Best Neighbor","client_self_model_accuracy_1":"Self"}

    create_data_per_measure("client_best_neighbor_model_accuracy_1")
    create_data_per_measure("client_self_model_accuracy_1")

    plot_nested_dict(for_graph[25],x_label="Iteration", y_label="Top 1 Accuracy", title="25 Clients")
    plot_nested_dict(for_graph[10],x_label="Iteration", y_label="Top 1 Accuracy", title="10 Clients")


    print()