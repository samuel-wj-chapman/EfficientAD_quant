import pandas as pd
from PIL import Image
import os
import torch
from torchvision.datasets import VisionDataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader

class OpenImagesPyTorch(VisionDataset):
    def __init__(self, img_dir, annotations_file, transforms=None):
        super().__init__(root=img_dir, transforms=transforms)
        self.annotations = pd.read_csv(annotations_file)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_id = self.annotations.iloc[idx]['ImageID']
        label = self.annotations.iloc[idx]['LabelName']
        img_path = os.path.join(self.img_dir, f'{img_id}.jpg')
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"File not found: {img_path}")
            return None

        if self.transforms:
            image = self.transforms(image)

        # Encode the label
        label = label_encoder.encode(label)
        return image, label

from collections import defaultdict

class LabelEncoder:
    def __init__(self):
        self.label_to_index = defaultdict(lambda: len(self.label_to_index))
        self.index_to_label = {}

    def encode(self, label):
        index = self.label_to_index[label]
        self.index_to_label[index] = label
        return index

    def decode(self, index):
        return self.index_to_label[index]

# Initialize the encoder globally
label_encoder = LabelEncoder()

def collate_fn(batch):
    batch = [x for x in batch if x is not None]  # Filter out None values
    if not batch:
        return torch.empty(0, 3, 256, 256), torch.empty(0, dtype=torch.long)
    images, labels = zip(*batch)  # Unpack images and labels
    images = torch.stack(images)  # Stack images into a single tensor
    labels = torch.tensor(labels, dtype=torch.long)  # Convert labels to a tensor
    return images, labels

# Define transformations
transform = Compose([
    Resize((256, 256)),  # Assuming images are not exactly 256x256
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = OpenImagesPyTorch(
    img_dir='dataset/downsampled-open-images-v4/256px/train-256',
    annotations_file='dataset/downsampled-open-images-v4/256px/train-annotations-human-imagelabels.csv',
    transforms=transform
)

validation_dataset = OpenImagesPyTorch(
    img_dir='dataset/downsampled-open-images-v4/256px/validation',
    annotations_file='dataset/downsampled-open-images-v4/256px/validation-annotations-human-imagelabels.csv',
    transforms=transform
)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
import torch
import torch.nn as nn
from common import get_pdn_small

class ExtendedPDNSmall(nn.Module):
    def __init__(self, num_classes=600):  # Adjust num_classes based on your dataset specifics
        super(ExtendedPDNSmall, self).__init__()
        self.base_model = get_pdn_small(out_channels=192)  # Load the base model
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

import torch.optim as optim

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ExtendedPDNSmall().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(num_epochs, model, train_loader):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in train_loader:
            if batch[0].size(0) == 0:
                print('batch_empty')
                continue  # Skip the loop if the batch is empty
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')
    print('Finished Training')

# Call the training function
# Call the training function
train_model(10, model, train_loader)

# Save only the base model
torch.save(model.base_model.state_dict(), 'pdn_small_open_images.pth')