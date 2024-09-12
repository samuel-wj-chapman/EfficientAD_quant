import os
import sys
import time
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch

def augment_and_save_images(root_dir):
    # Define the transformations
    transformations = {
        'rotate90': transforms.Lambda(lambda x: TF.rotate(x, 90)),
        'rotate180': transforms.Lambda(lambda x: TF.rotate(x, 180)),
        'rotate270': transforms.Lambda(lambda x: TF.rotate(x, 270)),
        'horizontal_flip': transforms.RandomHorizontalFlip(p=1),
        'vertical_flip': transforms.RandomVerticalFlip(p=1),
        'transpose': transforms.Compose([transforms.RandomVerticalFlip(p=1), transforms.RandomHorizontalFlip(p=1)]),
        'adjust_gamma_high': transforms.ColorJitter(brightness=0.5),
        'adjust_gamma_low': transforms.ColorJitter(brightness=1.5),
        'add_noise': transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.0005)
    }

    start_time = time.time()  # Start timing

    # Traverse the directory
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.png'):
                file_path = os.path.join(subdir, file)
                image = Image.open(file_path)
                image_tensor = transforms.ToTensor()(image)  # Convert to tensor for some transformations

                # Apply each transformation and save the new image
                for transform_name, transform in transformations.items():
                    new_image_tensor = transform(image_tensor)
                    new_image = transforms.ToPILImage()(new_image_tensor)  # Convert back to PIL Image to save
                    new_file_path = os.path.join(subdir, f"{os.path.splitext(file)[0]}_{transform_name}.png")
                    new_image.save(new_file_path)

    end_time = time.time()  # End timing
    print(f"Total time for augmentation: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python augmentor.py <directory>")
        sys.exit(1)
    directory = sys.argv[1]
    augment_and_save_images(directory)