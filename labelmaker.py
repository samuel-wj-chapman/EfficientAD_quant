import pandas as pd
import os
from tqdm import tqdm

# Paths to your annotations CSV files and the directories containing the images
train_annotations_file = 'dataset/downsampled-open-images-v4/256px/train-annotations-human-imagelabels.csv'
train_img_dir = 'dataset/downsampled-open-images-v4/256px/train-256'
validation_annotations_file = 'dataset/downsampled-open-images-v4/256px/validation-annotations-human-imagelabels.csv'
validation_img_dir = 'dataset/downsampled-open-images-v4/256px/validation'

# Load the annotations
train_annotations = pd.read_csv(train_annotations_file)
validation_annotations = pd.read_csv(validation_annotations_file)
def filter_images(annotations, img_dir, output_file):
    valid_entries = []
    chunk_size = 100000  # Define the size of each chunk
    count = 0

    for _, row in tqdm(annotations.iterrows(), total=annotations.shape[0], desc="Filtering images"):
        img_id = row['ImageID']
        img_path = os.path.join(img_dir, f'{img_id}.jpg')
        if os.path.exists(img_path):
            valid_entries.append(row)
            count += 1

        # Save every chunk_size entries
        if count % chunk_size == 0:
            pd.DataFrame(valid_entries).to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
            valid_entries = []  # Reset the list

    # Save any remaining entries
    if valid_entries:
        pd.DataFrame(valid_entries).to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)

# Usage of the modified function
filter_images(train_annotations, train_img_dir, 'filtered_train_annotations.csv')
filter_images(validation_annotations, validation_img_dir, 'filtered_validation_annotations.csv')

print("Filtered annotations have been saved in chunks.")