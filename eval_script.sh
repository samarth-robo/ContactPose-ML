#!/usr/bin/env bash
conda activate contactpose_ml

if [ $# -ne 4 ]; then
  echo "Usage: $0 config.ini split checkpoint.pth device_id"
  echo "Got $# arguments instead"
  exit -1
fi


for jd in `seq 0 0.05 0`; do
  for run in `seq 1 3`; do
    echo "############### Joint Dropout = ${jd}, Run = ${run}"
    python eval.py --save --device $4 --split $2 --config $1 --checkpoint $3 --joint_droprate ${jd} --output_filename_suffix run_${run}
  done
done
