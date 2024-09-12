#!/bin/bash

# Run the image augmentation script
python3 augmentor.py ./mvtec_anomaly_detection/bottle/train/

# Check if the previous command was successful
if [ $? -eq 0 ]; then
    # Run the anomaly detection training script
    python3 efficientad.py --dataset mvtec_ad --subdataset bottle
else
    echo "The image augmentation process failed."
    exit 1
fi