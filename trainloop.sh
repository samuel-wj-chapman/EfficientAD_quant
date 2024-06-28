#!/bin/bash

# Array of subdataset values
subdatasets=("screw" "tile" "toothbrush" "transistor" "wood" "zipper")

# Loop through each subdataset and run the Python script
for subdataset in "${subdatasets[@]}"; do
    echo "Running for subdataset: $subdataset"
    python3 efficientad.py --dataset mvtec_ad --subdataset "$subdataset"
done

echo "All subdatasets processed."



