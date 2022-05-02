#!/bin/bash

echo "-------------------------------------------------------"
echo "Step 1: make_triplet_txtds"
echo "-------------------------------------------------------"

python prep-make_triplet_txtds.py
mv prep_txt_ds/*.json ../txt_data/preprocessed/ -f

echo "-------------------------------------------------------"
echo "Step 2: cache preprocesed text file to numpy data"
echo "-------------------------------------------------------"
python prep-cache_txtds.py
