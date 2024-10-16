import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageNetDataset(Dataset):
    def __init__(self, img_dir, txt_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        with open(txt_file, 'r') as f:
            for line in f:
                
                parts = line.strip().split()
                if len(parts) != 2:
                    continue  
                path, label = parts
                image_path = os.path.join(img_dir, path)
                label_idx = int(label)
                self.image_paths.append(image_path)
                self.labels.append(label_idx)
                    
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Optionally, you can choose to skip this sample or return a placeholder image
            image = Image.new('RGB', (256, 256))  # Placeholder image
        if self.transform:
            image = self.transform(image)
        return image, label

# Define transformations (keeping normalization the same)
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to 256x256
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = ImageNetDataset(
    img_dir='/dataset/ILSVRC2012_img_train',
    txt_file='/dataset/train.txt',
    transform=transform
)

val_dataset = ImageNetDataset(
    img_dir='/dataset/ILSVRC2012_img_val',  # Adjust the path if necessary
    txt_file='/dataset/val.txt',
    transform=transform
)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)





import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
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

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ExtendedPDNSmall().to(device)
criterion = nn.CrossEntropyLoss()#label_smoothing=0.1
#optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

scheduler = CosineAnnealingLR(optimizer, T_max=100)  # Adjust T_max to your num_epochs


#scaler = GradScaler()
writer = SummaryWriter()


import os
import matplotlib.pyplot as plt

def save_sample_images(dataset, num_classes=20, output_dir='sample_images'):
    import random
    os.makedirs(output_dir, exist_ok=True)
    class_indices = set()
    samples_per_class = {}
    # Shuffle indices to get random samples
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    for idx in indices:
        _, label = dataset[idx]
        label = label.item() if isinstance(label, torch.Tensor) else label
        if label not in class_indices:
            class_indices.add(label)
            samples_per_class[label] = idx
        if len(class_indices) >= num_classes:
            break
    # Save images with class names
    for label, idx in samples_per_class.items():
        image, _ = dataset[idx]
        # Inverse transform to convert image back to PIL Image
        inv_transform = transforms.Compose([
            transforms.Normalize(
                mean=[-m/s for m, s in zip([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])],
                std=[1/s for s in [0.229, 0.224, 0.225]]
            ),
            transforms.ToPILImage()
        ])
        image_pil = inv_transform(image)
        #class_name = class_names[label]
        plt.figure()
        plt.imshow(image_pil)
        plt.title(label)
        plt.axis('off')
        # Save image
        image_filename = os.path.join(output_dir, f'class_{label}.png')
        plt.savefig(image_filename)
        plt.close()


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt



def train_one_epoch(epoch, model, optimizer, criterion, train_loader):
    model.train()
    running_loss = 0.0
    batch_count = 0
    for images, labels in train_loader:
        batch_count += 1
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if batch_count % 100 == 0:
            print(f'Epoch {epoch+1}, Iteration {batch_count}, Loss: {loss.item()}')
    average_loss = running_loss / batch_count
    return average_loss

def validate(model, criterion, val_loader):
    model.eval()
    val_loss = 0.0
    batch_count = 0
    with torch.no_grad():
        for images, labels in val_loader:
            batch_count += 1
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    val_loss /= batch_count
    return val_loss

def plot_losses(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_plot.png')  # Save the plot as an image file
    plt.show()

def train_model(num_epochs, model, train_loader, val_loader, patience=5):
    best_val_loss = float('inf')
    no_improve_epoch = 0
    torch.save(model.state_dict(), 'test_model.pth')
    
    # Lists to store losses
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(epoch, model, optimizer, criterion, train_loader)
        val_loss = validate(model, criterion, val_loader)
        scheduler.step()
        
        # Record losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Optionally, if using TensorBoard writer
        # writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epoch = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            no_improve_epoch += 1
            if no_improve_epoch >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        print(f'Epoch {epoch+1}, Training Loss: {train_loss}, Validation Loss: {val_loss}')
    
    print('Finished Training')
    
    # Optionally, close the writer if using TensorBoard
    # writer.close()
    
    # Plot the losses after training
    plot_losses(train_losses, val_losses)

# Call the training function
train_model(100, model, train_loader, val_loader)


