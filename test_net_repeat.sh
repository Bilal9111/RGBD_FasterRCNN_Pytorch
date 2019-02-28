#!/usr/bin/env bash


export SESSION=1
export CHECKPOINT=13783
export DECAY_STEP=5
export LEARNING_RATE=0.001




max=15
for i in `seq 1 $max`
do
    export EPOCH=$i
    python test_net.py --dataset kitti_voc --net res101 --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT --cuda
done
