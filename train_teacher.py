import pandas as pd
from PIL import Image
import os
import torch
from torchvision.datasets import VisionDataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
#annotations = pd.read_csv('filtered_train_annotations.csv')

# Count unique labels
#unique_labels = annotations['LabelName'].nunique()

#print(f"Number of unique labels: {unique_labels}")


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
#label_encoder = LabelEncoder()

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
'''
# Load datasets
train_dataset = OpenImagesPyTorch(
    img_dir='dataset/downsampled-open-images-v4/256px/train-256',
    annotations_file='filtered_train_annotations.csv',
    transforms=transform
)

validation_dataset = OpenImagesPyTorch(
    img_dir='dataset/downsampled-open-images-v4/256px/validation',
    annotations_file=  'filtered_validation_annotations.csv',
    transforms=transform
)
from PIL import Image
import numpy as np
import os
from PIL import Image
import numpy as np
import os

def save_images_with_label(dataset, target_label, num_images=20, save_dir='saved_images'):
    # Define the mean and std used for normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Create directory if it doesn't exist
    try:
        os.makedirs(save_dir, exist_ok=True)
    except Exception as e:
        print(f"Failed to create directory {save_dir}: {e}")
        return
    
    found_images = 0
    indices = []

    # Iterate through the dataset until we find enough images
    for i, (_, label) in enumerate(dataset):
        if label == target_label:
            indices.append(i)
            found_images += 1
            print(found_images)
            if found_images == num_images:
                break

    if found_images < num_images:
        print(f"Only found {found_images} images with label {target_label}.")

    for idx in indices:
        image, label = dataset[idx]
        # Reverse normalization
        image = image.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C) and to numpy array
        image = std * image + mean  # Reverse the normalization
        image = np.clip(image, 0, 1)  # Ensure the image range is valid
        
        # Convert to PIL image for saving
        image = (image * 255).astype(np.uint8)  # Scale to [0, 255]
        pil_image = Image.fromarray(image)
        
        # Convert label to string and sanitize it to remove characters that may not be valid in filenames
        safe_label = "".join([c for c in str(label) if c.isalnum() or c in "._-"])
        
        # Save image
        try:
            pil_image.save(f"{save_dir}/image_{idx}_label_{safe_label}.jpg")
        except Exception as e:
            print(f"Failed to save image {idx}: {e}")

# Assuming train_dataset is already loaded and is an instance of OpenImagesPyTorch
save_images_with_label(train_dataset, 3)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

'''
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Setup dataset
full_dataset = datasets.ImageFolder(root='datasets/places365/data_256', transform=transform)
#al_dataset = datasets.ImageFolder(root='~/datasets/places365/val', transform=transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
print('before')
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
# Setup data loader
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
#val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
total_batches = len(train_loader)
print(f"Total number of batches in the dataset: {total_batches}")


import torch
import torch.nn as nn
from common import get_pdn_small

class ExtendedPDNSmall(nn.Module):
    def __init__(self, num_classes=19786):  # Adjust num_classes based on your dataset specifics
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
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)



print('here')
# Training loop

def train_model(num_epochs, model, train_loader, val_loader, patience=3):
    import time
    model.train()
    best_val_loss = float('inf')
    no_improve_epoch = 0

    for epoch in range(num_epochs):
        start_time = time.time()
        running_loss = 0.0
        batch_count = 0
        for images, labels in train_loader:
            if images.size(0) == 0:
                print('Empty batch skipped')
                continue
            batch_count += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(f'Epoch {epoch+1}, Iteration {batch_count}, Loss: {loss.item()}')
        # Validation phase
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)


        average_loss = running_loss / batch_count
        print(f'Epoch {epoch+1}, Average Loss: {average_loss}')
        model_filename = f'pdn_small_open_images_epoch_{epoch+1}_loss_{average_loss:.4f}_vloss_{val_loss:.4f}.pth'
        torch.save(model.base_model.state_dict(), model_filename)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epoch = 0
        else:
            no_improve_epoch += 1
            if no_improve_epoch >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        print(f'Epoch {epoch+1}, Training Loss: {running_loss / batch_count}, Validation Loss: {val_loss}')

    print('Finished Training')

train_model(50, model, train_loader, val_loader)

# Save only the base model
torch.save(model.base_model.state_dict(), 'finalmodelpdn_small_open_images.pth')