import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Paths
anomaly_maps_dir = 'output/1/anomaly_maps/mvtec_ad'
original_images_dir = '/dataset2/mvtec_anomaly_detection'
output_dir = 'processed_images'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Font for overlaying text
# Use a default font that's likely to be available on most systems
font = ImageFont.load_default().font_variant(size=40)  # Adjust the size as needed


# Process each sub-dataset
for sub_dataset in os.listdir(anomaly_maps_dir):
    sub_dataset_path = os.path.join(anomaly_maps_dir, sub_dataset)
    if not os.path.isdir(sub_dataset_path):
        continue

    # Prepare list to hold comparison images
    comparison_images = []

    # Adjusted path to include 'test' directory if needed
    test_dir_path = os.path.join(sub_dataset_path, 'test')
    if not os.path.isdir(test_dir_path):
        test_dir_path = sub_dataset_path  # Fallback if 'test' directory doesn't exist

    # Walk through all subdirectories and files
    for root, dirs, files in os.walk(test_dir_path):
        for file_name in files:
            if file_name.lower().endswith('.tiff') or file_name.lower().endswith('.tif'):
                tiff_path = os.path.join(root, file_name)

                # Load the tiff image
                anomaly_img = Image.open(tiff_path)
                anomaly_array = np.array(anomaly_img).astype(np.float32)
                max_value = anomaly_array.max()
                # Multiply all values by 255
                anomaly_array *= 30

                
                

                # Convert array back to image
                anomaly_array = np.clip(anomaly_array, 0, 255).astype(np.uint8)
                anomaly_img = Image.fromarray(anomaly_array).convert('RGB')

                # Build the relative path to get the category
                relative_path = os.path.relpath(root, anomaly_maps_dir)
                category_path = os.path.join(relative_path, file_name)

                # Load the corresponding original image
                # Construct the path to the original image based on the category
                base_name = os.path.splitext(file_name)[0]
                original_img_path = os.path.join(
                    original_images_dir, relative_path, base_name + '.png'  # Adjust extension if necessary
                )
                if not os.path.exists(original_img_path):
                    print(f'Original image not found: {original_img_path}')
                    continue
                original_img = Image.open(original_img_path).convert('RGB')

                # Create a comparison image (side by side)
                width, height = anomaly_img.size
                new_height = height + 100  # Extra space for text
                comparison_img = Image.new('RGB', (width * 2, new_height), (0, 0, 0))
                comparison_img.paste(original_img.resize((width, height)), (0, 0))
                comparison_img.paste(anomaly_img.resize((width, height)), (width, 0))

                # Add category path and max value below the images
                draw = ImageDraw.Draw(comparison_img)
                title_text = f'Category: {relative_path}'
                max_value_text = f'Max Value: {max_value:.2f}'
                text_position = (10, height + 5)
                draw.text(text_position, title_text, fill=(255, 255, 255), font=font)
                draw.text((10, height + 40), max_value_text, fill=(255, 255, 255), font=font)

                # Add to list
                comparison_images.append(comparison_img)

    # Arrange all comparison images into a single image
    if comparison_images:
        num_images = len(comparison_images)
        images_per_row = 3
        num_rows = (num_images + images_per_row - 1) // images_per_row
        img_width, img_height = comparison_images[0].size

        composite_image = Image.new('RGB', (images_per_row * img_width, num_rows * img_height))

        for idx, comp_img in enumerate(comparison_images):
            row = idx // images_per_row
            col = idx % images_per_row
            composite_image.paste(comp_img, (col * img_width, row * img_height))

        # Save the composite image
        composite_image_path = os.path.join(output_dir, f'{sub_dataset}_comparison.png')
        composite_image.save(composite_image_path)
        print(f'Composite image saved: {composite_image_path}')
