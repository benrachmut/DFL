import torch.nn as nn
import torchvision


class VGGServer(nn.Module):
    def __init__(self, num_classes, num_clusters=1):
        super(VGGServer, self).__init__()
        self.vgg = torchvision.models.vgg16(weights=None)  # No pre-trained weights
        self.vgg.classifier[6] = nn.Linear(4096, num_classes)  # Adjust for final output

        self.num_clusters = num_clusters

        if num_clusters == 1:
            # Single head case
            self.head = nn.Linear(num_classes, num_classes)
        else:
            # Multi-head with deeper layers
            self.heads = nn.ModuleDict({
                f"head_{i}": nn.Sequential(
                    nn.Linear(num_classes, 512), nn.ReLU(),
                    nn.Linear(512, 256), nn.ReLU(),
                    nn.Linear(256, num_classes)
                ) for i in range(num_clusters)
            })

    def forward(self, x, cluster_id=None):
        # Resize input to match VGG's expected input size
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.vgg(x)  # Backbone output

        if self.num_clusters == 1:
            return self.head(x)  # Single-head case

        if cluster_id is not None:
            return self.heads[f"head_{cluster_id}"](x)  # Select head by cluster_id

        # If no cluster_id is given, return outputs from all heads
        return {f"head_{i}": head(x) for i, head in self.heads.items()}


class AlexNet(nn.Module):
    def __init__(self, num_classes, num_clusters=1):
        super(AlexNet, self).__init__()

        # Backbone (shared layers)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 4096), nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout()
        )

        # If there's only 1 head, keep it simple (like original)
        if num_clusters == 1:
            self.head = nn.Linear(4096, num_classes)  # Single head
        else:
            # Multi-head with deeper structures
            self.heads = nn.ModuleDict({
                f"head_{i}": nn.Sequential(
                    nn.Linear(4096, 2048), nn.ReLU(),
                    nn.Linear(2048, 1024), nn.ReLU(),
                    nn.Linear(1024, num_classes)
                ) for i in range(num_clusters)
            })

        self.num_clusters = num_clusters

    def forward(self, x, cluster_id=None):
        x = self.backbone(x)

        if self.num_clusters == 1:
            return self.head(x)  # Single-head case

        if cluster_id is not None:
            return self.heads[f"head_{cluster_id}"](x)  # Select head by cluster_id

        # If no cluster_id is given, return outputs from all heads
        return {f"head_{i}": head(x) for i, head in self.heads.items()}
