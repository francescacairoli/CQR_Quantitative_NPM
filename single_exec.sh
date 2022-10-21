#!/bin/bash

############
# settings #
############

MODEL_PREFIX="MRH" 
MODEL_DIM=4 

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
LOGS="out/logs/"
mkdir -p $LOGS
OUT="${LOGS}${DATE}_${TIME}_out.txt"


#######
# run #
#######

for i in 0 1 2 3
do
	echo i: $i
	python stoch_run_cqr.py --property_idx $i --model_prefix $MODEL_PREFIX --model_dim $MODEL_DIM >> $OUT 2>&1
done