import torch
import torch.nn as nn
from common import get_pdn_small

class ExtendedPDNSmall(nn.Module):
    def __init__(self, num_classes=1000):
        super(ExtendedPDNSmall, self).__init__()
        self.base_model = get_pdn_small(out_channels=192)  # Use the improved base model
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, 512),  # Reduced size
            nn.BatchNorm1d(512),  # Added BatchNorm
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1024),  # Reduced size
            nn.BatchNorm1d(1024),  # Added BatchNorm
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

# Ensure you're using the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the ExtendedPDNSmall model
model = ExtendedPDNSmall().to(device)

# Load the trained model's state dict
model.load_state_dict(torch.load('best_model_2.7loss.pth', map_location=device))

print("Loaded trained ExtendedPDNSmall model.")

base_model = model.base_model

# Save the base_model's state_dict
torch.save(base_model.state_dict(), 'pdn_tiny_base_model_teacher.pth')

print("Saved base_model's state_dict as 'pdn_small_base_model.pth'.")
