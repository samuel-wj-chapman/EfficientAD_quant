#!/bin/bash

# Create directory and navigate into it
mkdir mvtec_anomaly_detection
cd mvtec_anomaly_detection

# Download the dataset and extract it
wget https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz
tar -xvf mvtec_anomaly_detection.tar.xz
cd ..

# Download the evaluation code, extract it, and clean up the tar file
wget https://www.mydrive.ch/shares/60736/698155e0e6d0467c4ff6203b16a31dc9/download/439517473-1665667812/mvtec_ad_evaluation.tar.xz
tar -xvf mvtec_ad_evaluation.tar.xz
rm mvtec_ad_evaluation.tar.xz