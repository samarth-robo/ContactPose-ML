#!/usr/bin/env bash
conda activate contactpose_ml

if [ $# -ne 4 ]; then
  echo "Usage: $0 config.ini split checkpoint.pth device_id"
  echo "Got $# arguments instead"
  exit -1
fi


python eval.py --save --device $4 --split $2 --config $1 --checkpoint $3 --joint_droprate 0.15 --output_filename_suffix run_1
