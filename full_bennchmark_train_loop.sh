#!/bin/bash
objects=("bottle" "cable" "capsule" "carpet" "grid" "hazelnut" "leather" "metal_nut" "pill" "screw" "tile" "toothbrush" "transistor" "wood" "zipper")
#objects=("screw" "tile" "toothbrush" "transistor" "wood" "zipper")

for obj in "${objects[@]}"
do
   python3 efficientadanom_only.py --subdataset "$obj"
done
