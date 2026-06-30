#!/bin/bash

base_dataset_dir="/kaggle/working/parrot_dataset"
base_data_dir="runs/aligner"
config_file="utils/aligner/aligner_train_config.yaml"

for speaker in "$base_dataset_dir"/*; do
  speaker_name=$(basename "$speaker")

  sed -i "s|dataset_dir:.*|dataset_dir: ${base_dataset_dir}/${speaker_name}|g" "$config_file"
  sed -i "s|data_dir:.*|data_dir: ${base_data_dir}/${speaker_name}|g" "$config_file"

  python utils/aligner/character_preprocess.py --config "$config_file"
  python utils/aligner/train.py --config "$config_file"
  python utils/aligner/extract_durations.py --config "$config_file"

done
