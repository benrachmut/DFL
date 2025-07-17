from torch.utils.data import DataLoader
import torch.nn.functional as F

from config_ import *
from NN import *
from abc import ABC, abstractmethod

def get_client_hub_model():
    if ec.client_hub_net == NetType.AlexNet:
        ec.update_alexNet()
        return AlexNet(num_classes=ec.num_classes).to(device)
    if ec.client_hub_net == NetType.VGG:
        ec.update_vgg()
        return VGGServer(num_classes=ec.num_classes).to(device)

def get_client_non_hub_model():
    if ec.client_non_hub_net == NetType.AlexNet:
        ec.update_alexNet()
        return AlexNet(num_classes=ec.num_classes).to(device)
    if ec.client_non_hub_net == NetType.VGG:
        ec.update_vgg()
        return VGGServer(num_classes=ec.num_classes).to(device)




class RecordData:
    def __init__(self, clients):#loss_measures,accuracy_measures,accuracy_pl_measures,accuracy_measures_k,accuracy_pl_measures_k):
        self.summary = ec.to_dict()


        ########
        self.client_loss_train = {}
        self.client_loss_test = {}
        #########
        self.client_self_model_accuracy_1 = {}
        self.client_best_neighbor_model_accuracy_1 = {}
        self.client_self_model_accuracy_5 = {}
        self.client_best_neighbor_model_accuracy_5 = {}
        self.client_self_model_accuracy_10= {}
        self.client_best_neighbor_model_accuracy_10= {}
        #########
        self.strong_client_loss_train = {}
        self.strong_client_loss_test = {}
        #########
        self.strong_client_self_model_accuracy_1 = {}
        self.strong_client_best_neighbor_model_accuracy_5 = {}


        for client in clients:
            ########
            self.client_loss_train[client.id_] = client.loss_train
            self.client_loss_test[client.id_] = client.loss_test
            #########
            self.client_self_model_accuracy_1[client.id_] = client.self_model_accuracy_1
            self.client_best_neighbor_model_accuracy_1[client.id_] = client.best_neighbor_model_accuracy_1
            self.client_self_model_accuracy_5[client.id_] = client.self_model_accuracy_5
            self.client_best_neighbor_model_accuracy_5[client.id_] = client.best_neighbor_model_accuracy_5
            self.client_self_model_accuracy_10[client.id_] =  client.self_model_accuracy_10
            self.client_best_neighbor_model_accuracy_10[client.id_] =  client.best_neighbor_model_accuracy_10
            if client.id_ in ec.selected_hubs:
                self.strong_client_loss_train[client.id_]=client.loss_train
                self.strong_client_loss_test[client.id_]=client.loss_test
                self.strong_client_self_model_accuracy_1[client.id_]=client.self_model_accuracy_1
                self.strong_client_best_neighbor_model_accuracy_5[client.id_]=client.self_model_accuracy_5
                self.strong_client_self_model_accuracy_10 = {}

class Client(ABC):
    def __init__(self, client_id, data_dict):
        self.id_ = client_id
        self.seed = ec.num_run
        for key, value in data_dict.items():
            setattr(self, key, value)
        self.model = self.get_model()
        self.epoch_count = 0
        self.data_to_send = None
        self.received_data = {}

        self.loss_train = {}
        self.loss_test = {}
        #########
        self.self_model_accuracy_1 = {}
        self.best_neighbor_model_accuracy_1 = {}
        self.self_model_accuracy_5= {}
        self.best_neighbor_model_accuracy_5 = {}
        self.self_model_accuracy_10= {}
        self.best_neighbor_model_accuracy_10 = {}

    def get_model(self):
        if self.id_ in ec.selected_hubs:
            return get_client_hub_model()
        else:
            return get_client_non_hub_model()

    def initialize_weights(self, layer):
        """Initialize weights for the model layers."""
        # Ensure the seed is within the valid range for PyTorch
        safe_seed = int(self.seed) % (2 ** 31)

        # Update the seed with experiment_config.seed_num
        self.seed = (self.seed + 1)
        torch.manual_seed(self.seed)  # For PyTorch
        torch.cuda.manual_seed(self.seed)  # For CUDA (if using GPU)
        torch.cuda.manual_seed_all(self.seed)  # For multi-GPU


        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)



    def __repr__(self):
        attrs = ', '.join(f"{k}={type(v).__name__}" for k, v in self.__dict__.items() if k != 'id_')
        return f"Client(id={self.id_}, {attrs})"


    def train(self,mean_pseudo_labels):

        print(f"Mean pseudo-labels shape: {mean_pseudo_labels.shape}")  # Should be (num_data_points, num_classes)

        print(f"*** {self.__str__()} train ***")
        server_loader = DataLoader(self.UL, batch_size=ec.batch_size, shuffle=False, num_workers=0,
                                   drop_last=True)
        #server_loader = DataLoader(self.global_data, batch_size=experiment_config.batch_size, shuffle=False,
        #                           num_workers=0)
        #print(1)
        self.model.train()
        criterion = nn.KLDivLoss(reduction='batchmean')
        optimizer = torch.optim.Adam( self.model.parameters(), lr=ec.learning_rate_train)
        #optimizer = torch.optim.Adam(self.model.parameters(), lr=experiment_config.learning_rate_train_c,
        #                             weight_decay=1e-4)

        pseudo_targets_all = mean_pseudo_labels.to(device)

        for epoch in range(ec.epochs_num):
            #print(2)

            self.epoch_count += 1
            epoch_loss = 0

            for batch_idx, (inputs, _) in enumerate(server_loader):
                #print(batch_idx)

                inputs = inputs.to(device)
                optimizer.zero_grad()

                outputs =  self.model(inputs)
                # Check for NaN or Inf in outputs

                # Convert model outputs to log probabilities
                outputs_prob = F.log_softmax(outputs, dim=1)
                # Slice pseudo_targets to match the input batch size
                start_idx = batch_idx * ec.batch_size
                end_idx = start_idx + inputs.size(0)
                pseudo_targets = pseudo_targets_all[start_idx:end_idx].to(device)

                # Check if pseudo_targets size matches the input batch size
                if pseudo_targets.size(0) != inputs.size(0):
                    print(
                        f"Skipping batch {batch_idx}: Expected pseudo target size {inputs.size(0)}, got {pseudo_targets.size(0)}")
                    continue  # Skip the rest of the loop for this batch

                # Check for NaN or Inf in pseudo targets
                if torch.isnan(pseudo_targets).any() or torch.isinf(pseudo_targets).any():
                    print(f"NaN or Inf found in pseudo targets at batch {batch_idx}: {pseudo_targets}")
                    continue

                # Normalize pseudo targets to sum to 1
                pseudo_targets = F.softmax(pseudo_targets, dim=1)

                # Calculate the loss
                loss = criterion(outputs_prob, pseudo_targets)

                # Check if the loss is NaN or Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN or Inf loss encountered at batch {batch_idx}: {loss}")
                    continue

                loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(server_loader)
            print(f"Epoch [{epoch + 1}/{ec.epochs_num}], Loss: {avg_loss:.4f}")

        #self.weights =self.model.state_dict()
        return avg_loss


    def fine_tune(self):
        print("*** " + self.__str__() + " fine-tune ***")

        # Load the weights into the model
        #if self.weights is  None:
        #    self.model.apply(self.initialize_weights)
        #else:
        #    self.model.load_state_dict(self.weights)

        # Create a DataLoader for the local data
        fine_tune_loader = DataLoader(self.train_data, batch_size=ec.batch_size, shuffle=True)
        self.model.train()  # Set the model to training mode

        # Define loss function and optimizer

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=ec.learning_rate_fine_tune)

        epochs = ec.epochs_num
        for epoch in range(epochs):
            self.epoch_count += 1
            epoch_loss = 0
            for inputs, targets in fine_tune_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = criterion(outputs, targets)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            result_to_print = epoch_loss / len(fine_tune_loader)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {result_to_print:.4f}")
        #self.weights = self.model.state_dict()self.weights = self.model.state_dict()

        return  result_to_print

    def evaluate_test_loss(self):
        self.model.eval()  # Set the model to evaluation mode
        test_loader = DataLoader(self.test_data, batch_size=ec.batch_size, shuffle=False)
        criterion = nn.CrossEntropyLoss()
        total_loss = 0

        with torch.no_grad():  # Disable gradient tracking for evaluation
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()

        average_loss = total_loss / len(test_loader)
        print(f"Test Loss: {average_loss:.4f}")
        return average_loss

    def evaluate(self, model=None):
        if model is None:
            model = self.model
    #    print("*** Generating Pseudo-Labels with Probabilities ***")

        # Create a DataLoader for the global data
        global_data_loader = DataLoader(self.UL, batch_size=ec.batch_size, shuffle=False)

        model.eval()  # Set the model to evaluation mode

        all_probs = []  # List to store the softmax probabilities
        with torch.no_grad():  # Disable gradient computation
            for inputs, _ in global_data_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)  # Forward pass

                # Apply softmax to get the class probabilities
                probs = F.softmax(outputs, dim=1)  # Apply softmax along the class dimension

                all_probs.append(probs.cpu())  # Store the probabilities on CPU

        # Concatenate all probabilities into a single tensor (2D matrix)
        all_probs = torch.cat(all_probs, dim=0)

       #print(f"Shape of the 2D pseudo-label matrix: {all_probs.shape}")
        return all_probs

    @abstractmethod
    def initialize(self):pass

    @abstractmethod
    def compute(self,t):pass

    def evaluate_accuracy(self, data_, model=None, k=1, cluster_id=None):
        if model is None:
            model = self.model

        """
        Evaluate the top-k accuracy of the model on the given dataset.

        Args:
            data_ (torch.utils.data.Dataset): The dataset to evaluate.
            model (torch.nn.Module): The model to evaluate.
            k (int): Top-k accuracy.
            cluster_id (int or None): Used if model has multiple heads.

        Returns:
            float: Top-k accuracy (%) on the dataset.
        """
        model.eval()
        correct = 0
        total = 0

        test_loader = DataLoader(data_, batch_size=ec.batch_size, shuffle=False)

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs, cluster_id=cluster_id)

                # Ensure outputs has correct dimensions
                if outputs.dim() == 1:
                    outputs = outputs.unsqueeze(0)  # make it [1, num_classes]

                # Top-k predictions (returns both values and indices)
                if ec.num_classes < k:
                    return 0
                else:
                    _, topk_preds = outputs.topk(k, dim=1)
                # Check if the correct label is in the top-k predictions
                correct += (topk_preds == targets.unsqueeze(1)).any(dim=1).sum().item()
                total += targets.size(0)

        accuracy = 100 * correct / total if total > 0 else 0.0
        print(f"Top-{k} Accuracy for cluster {cluster_id if cluster_id is not None else 'default'}: {accuracy:.2f}%")
        return accuracy

class Client_DMAPLE(Client):
    def __init__(self, client_id, data_dict):
        Client.__init__(self, client_id, data_dict)
        self.aggregated_pl = None
        self.neighbors = ec.neighbors_dict[client_id]
    def collect_data(self,t):
        self.self_model_accuracy_1[0] = self.evaluate_accuracy(self.test_data, k=1)
        self.self_model_accuracy_5[0] = self.evaluate_accuracy(self.test_data, k=5)
        self.self_model_accuracy_10[0] = self.evaluate_accuracy(self.test_data, k=10)
        self.loss_test[0] = self.evaluate_test_loss()

    def initialize(self):
        self.loss_train[0] = self.fine_tune()
        self.data_to_send = self.evaluate()
        self.collect_data(0)



    def compute(self,t):
        self.train(self.aggregated_pl)
        self.loss_train[t] = self.fine_tune()
        self.data_to_send = self.evaluate()
        self.collect_data(t)




    def select_confident_pseudo_labels(self, cluster_pseudo_labels):
        """
        Select pseudo-labels from the cluster with the highest confidence for each data point.

        Args:
            cluster_pseudo_labels (list of torch.Tensor): List of tensors where each tensor contains pseudo-labels
                                                          from a cluster with shape [num_data_points, num_classes].

        Returns:
            torch.Tensor: A tensor containing the selected pseudo-labels of shape [num_data_points, num_classes].
        """
        num_data_points = cluster_pseudo_labels[0].size(0)

        # Store the maximum confidence and the corresponding cluster index for each data point
        max_confidences = torch.zeros(num_data_points, device=cluster_pseudo_labels[0].device)
        selected_labels = torch.zeros_like(cluster_pseudo_labels[0])

        for cluster_idx, pseudo_labels in enumerate(cluster_pseudo_labels):
            # Compute the max confidence for the current cluster
            cluster_max_confidences, _ = torch.max(pseudo_labels, dim=1)

            # Update selected labels where the current cluster has higher confidence
            mask = cluster_max_confidences > max_confidences
            max_confidences[mask] = cluster_max_confidences[mask]
            selected_labels[mask] = pseudo_labels[mask]

        return selected_labels

    def digest_received_data(self):
        pseudo_labels_list = list(self.received_data.values())
        self.aggregated_pl = self.select_confident_pseudo_labels(pseudo_labels_list)