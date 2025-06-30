#!/bin/bash
python generate_comparison_explanations.py --comparison_config "./comparison_configs/mnist_835_experiment/mnist_835_experiment_ckpt=epoch=1-step=184.json" --comparison_output_root outputs/mnist_835_experiment --save_m0_representation  --save_m1_representation
