#!/bin/bash

# Define the list of objects
objects=("bottle" "cable" "capsule" "carpet" "grid" "hazelnut" "leather" "metal_nut" "pill" "screw" "tile" "toothbrush" "transistor" "wood" "zipper")

# Initialize sum variables
sum_au_pro=0
sum_au_roc=0
count=0

# Loop through each object and run the evaluation
for object in "${objects[@]}"
do
    echo "Evaluating $object..."
    # Capture the output of the Python script
    output=$(python3 mvtec_ad_evaluation/evaluate_experiment.py --dataset_base_dir './mvtec_anomaly_detection/' --anomaly_maps_dir "./output/1/anomaly_maps/mvtec_ad/" --output_dir "./output/1/metrics/mvtec_ad/" --evaluated_objects "$object")
    
    # Print the output for debugging
    echo "Output for $object:"
    echo "$output"

    # Extract AU-PRO and AU-ROC values using grep and awk, ensuring only numbers are captured
    au_pro=$(echo "$output" | grep "AU-PRO" | awk -F'0.3): ' '{print $2}' | sed 's/[^0-9.]//g')
    au_roc=$(echo "$output" | grep "Image-level classification AU-ROC" | awk -F': ' '{print $2}' | sed 's/[^0-9.]//g')
    
    # Print the extracted values for debugging
    echo "Extracted AU-PRO for $object: $au_pro"
    echo "Extracted AU-ROC for $object: $au_roc"

    # Check if the variables are not empty and numeric
    if [[ ! -z "$au_pro" && "$au_pro" =~ ^[0-9]+(\.[0-9]+)?$ ]] && [[ ! -z "$au_roc" && "$au_roc" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        # Add to the sum and increment count
        sum_au_pro=$(echo "$sum_au_pro + $au_pro" | bc)
        sum_au_roc=$(echo "$sum_au_roc + $au_roc" | bc)
        ((count++))
    else
        echo "Failed to parse metrics for $object"
    fi
done

# Calculate averages if count is not zero
if [ $count -ne 0 ]; then
    average_au_pro=$(echo "$sum_au_pro / $count" | bc -l)
    average_au_roc=$(echo "$sum_au_roc / $count" | bc -l)

    # Print averages
    echo "Average AU-PRO (FPR limit: 0.3): $average_au_pro"
    echo "Average Image-level classification AU-ROC: $average_au_roc"
else
    echo "No valid data to calculate averages."
fi