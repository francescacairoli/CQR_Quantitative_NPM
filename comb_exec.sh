#!/bin/bash

############
# settings #
############

# model dim = 2 nb comb = 1
# model dim = 4 nb comb = 6
# model dim = 8 nb comb = 28
MODEL_PREFIX="MRH" 
MODEL_DIM=8 

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
LOGS="out/logs/"
mkdir -p $LOGS
OUT="${LOGS}${DATE}_${TIME}_out.txt"

#######
# run #
#######

for i in 0 1 2 3 4 5 6 # 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27
do
	echo i: $i
	python comb_stoch_run_cqr.py --qr_training_flag False --comb_calibr_flag True --comb_idx $i --model_prefix $MODEL_PREFIX --model_dim $MODEL_DIM >> $OUT 2>&1
done