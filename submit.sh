#!/usr/bin/env bash

# CUDA_VISIBLE_DEVICES=$1 nohup python keras_check.py ntrack $2 &> "$2"_ntrack.log && \
train_both_models() {
    CUDA_VISIBLE_DEVICES=$1 nohup python keras_check.py rnn $2 &> "$2"_rnn.log
}


# cee0f39, de41eff, 83b91b4, 468a1e4, 0d64b32, e800916, f3e9092, 832aabb, ca2044e

# train_both_models 0 cee0f39 &
train_both_models 1 de41eff &
train_both_models 2 83b91b4 &
train_both_models 3 468a1e4 &
# train_both_models 0 0d64b32 &
# train_both_models 1 e800916 &
# train_both_models 2 f3e9092 &
# train_both_models 3 832aabb &
# train_both_models 0 ca2044e &