import pandas as pd
import os
from tqdm import tqdm

# Paths to your annotations CSV files and the directories containing the images
train_annotations_file = 'filtered_train_annotations.csv'#'dataset/downsampled-open-images-v4/256px/train-annotations-human-imagelabels.csv'
train_img_dir = 'dataset/downsampled-open-images-v4/256px/train-256'
validation_annotations_file = 'filtered_validation_annotations.csv'#'dataset/downsampled-open-images-v4/256px/validation-annotations-human-imagelabels.csv'
validation_img_dir = 'dataset/downsampled-open-images-v4/256px/validation'

# Load the annotations
train_annotations = pd.read_csv(train_annotations_file)
validation_annotations = pd.read_csv(validation_annotations_file)

# Function to check the existence of image files
def check_images_existence(annotations, img_dir):
    found_count = 0
    not_found_count = 0

    for _, row in tqdm(annotations.iterrows(), total=annotations.shape[0], desc="Checking images"):
        img_id = row['ImageID']
        img_path = os.path.join(img_dir, f'{img_id}.jpg')
        if os.path.exists(img_path):
            found_count += 1
        else:
            not_found_count += 1

    return found_count, not_found_count

# Check the images in the training and validation datasets
train_found, train_not_found = check_images_existence(train_annotations, train_img_dir)
validation_found, validation_not_found = check_images_existence(validation_annotations, validation_img_dir)

# Print the results
print(f"Training images found: {train_found}, Training images not found: {train_not_found}")
print(f"Validation images found: {validation_found}, Validation images not found: {validation_not_found}")