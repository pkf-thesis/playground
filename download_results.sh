#!/bin/bash
scp -r -oProxyJump=$1@130.226.142.166 $1@10.1.1.121:/usr/local/share/FKP/results/predictions_SampleCNN_deep_resnet_mtat_1.6e-05.npy ../mixed_pooling/