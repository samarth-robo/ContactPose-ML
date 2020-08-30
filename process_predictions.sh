#!/usr/bin/env bash
conda activate contactpose_ml

if [ $# -ne 1 ]; then
  echo "Usage: ./$0 base_dir"
  exit -1
fi

BASE_DIR=$1

for exp in {pointnet_split_objects_simple-joints,mlp_split_objects_simple-joints,pointnet_split_participants_simple-joints,mlp_split_participants_simple-joints}; do
  for drop in `seq 0 0.05 0.0`; do
    pred=$BASE_DIR/${exp}/predictions_joint_droprate=${drop}_run_1.pkl
    echo ${pred}
    python utils/process_predictions.py --pred ${pred} --n_runs 3
  done
done
