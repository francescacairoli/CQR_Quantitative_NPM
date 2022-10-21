#!/bin/bash

############
# settings #
############

# model dim = 2 nb comb = 1
# model dim = 4 nb comb = 6
# model dim = 8 nb comb = 28
MODEL_PREFIX="MRH" 
MODEL_DIM=4 

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
LOGS="out/logs/"
mkdir -p $LOGS
OUT="${LOGS}${DATE}_${TIME}_out.txt"

NB_COMB=6
#######
# run #
#######
counter=0
for i in 0 1 2 3 4 5 
do
	echo i: $i
	python comb_stoch_run_cqr.py --comb_calibr_flag False --comb_idx $i --model_prefix $MODEL_PREFIX --model_dim $MODEL_DIM >> $OUT 2>&1
done