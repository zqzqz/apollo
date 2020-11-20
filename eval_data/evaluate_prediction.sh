#!/bin/bash
EVAL_DIR="/apollo/eval_data"

for d in {1..10}
do
    for t in {1..12}
    do
        sed -i "9s/.*/${d},${t},0/" ${EVAL_DIR}/perturbation.txt
        sed -i "10s/.*/${d},${t},0/" ${EVAL_DIR}/perturbation.txt
        timeout 30 /apollo/bazel-bin/modules/prediction/evaluate_prediction 1>>${EVAL_DIR}/output.tmp 2>>${EVAL_DIR}/log.tmp
        mv ${EVAL_DIR}/trajectories/predict.pb.txt ${EVAL_DIR}/trajectories/${d}_${t}_0_1.pb.txt
    done
done
