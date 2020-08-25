#!/usr/bin/env bash
conda activate contactpose_ml

if [ $# -ne 1 ]; then
  echo "Usage: ./$0 base_dir"
  exit -1
fi

BASE_DIR=$1

for exp in {pointnet_split_objects_mesh,pointnet_split_participants_mesh,mlp_split_objects_mesh,mlp_split_participants_mesh}; do
  for drop in `seq 0 0.05 0`; do
    pred=$BASE_DIR/${exp}/predictions_joint_droprate=${drop}_run_1.pkl
    echo ${pred}
    python utils/process_predictions.py --pred ${pred} --n_runs 3
  done
done
