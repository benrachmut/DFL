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



class Client(ABC):
    def __init__(self, customer_id, data_dict):
        self.id_ = customer_id
        self.seed = ec.num_run
        for key, value in data_dict.items():
            setattr(self, key, value)
        self.model = self.get_model()
        self.epoch_count = 0

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

    @abstractmethod
    def initialize(self):pass

    def __repr__(self):
        attrs = ', '.join(f"{k}={type(v).__name__}" for k, v in self.__dict__.items() if k != 'id_')
        return f"Client(id={self.id_}, {attrs})"


    def train(self,mean_pseudo_labels):

        print(f"Mean pseudo-labels shape: {mean_pseudo_labels.shape}")  # Should be (num_data_points, num_classes)

        print(f"*** {self.__str__()} train ***")
        server_loader = DataLoader(self.global_data, batch_size=ec.batch_size, shuffle=False, num_workers=0,
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


class Client_DMAPLE(Client):
    def __init__(self,customer_id, data_dict):
        Client.__init__(self,customer_id, data_dict)