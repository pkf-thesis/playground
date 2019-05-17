#!/bin/bash
scp -r -oProxyJump=$1@130.226.142.166 $1@10.1.1.121:/usr/local/share/FKP/results/best_weights_mixed_region_8e-05.hdf5 ../