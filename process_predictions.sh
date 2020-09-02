#!/usr/bin/env bash
conda activate contactpose_ml

if [ $# -ne 1 ]; then
  echo "Usage: ./$0 base_dir"
  exit -1
fi

BASE_DIR=$1
# dummy
drop=0.15

for exp in {images_split_images_objects_1view,images_split_images_participants_1view,images_split_images_objects_3view,images_split_images_participants_3view}; do
  pred=$BASE_DIR/${exp}/predictions_joint_droprate=${drop}_run_1.pkl
  echo ${pred}
  python utils/process_predictions.py --n_runs 1 --no_rotations --pred ${pred}
done
