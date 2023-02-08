#!/bin/bash

############
# settings #
############

LOGS="out/logs/"
mkdir -p $LOGS



DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
OUT="${LOGS}${DATE}_${TIME}_out.txt"

MODEL_PREFIX="GRN" 
MODEL_DIM=2

for i in 0
do
	echo $MODEL_PREFIX$MODEL_DIM i: $i
	python comb_stoch_run_cqr.py --qr_training_flag False --comb_calibr_flag True --comb_idx $i --model_prefix $MODEL_PREFIX --model_dim $MODEL_DIM >> $OUT 2>&1
	python comb_stoch_run_cqr.py --qr_training_flag False --comb_calibr_flag False --comb_idx $i --model_prefix $MODEL_PREFIX --model_dim $MODEL_DIM >> $OUT 2>&1
done

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
OUT="${LOGS}${DATE}_${TIME}_out.txt"

MODEL_PREFIX="MRH" 
MODEL_DIM=2

for i in 0
do
	echo $MODEL_PREFIX$MODEL_DIM i: $i
	python comb_stoch_run_cqr.py --qr_training_flag False --comb_calibr_flag True --comb_idx $i --model_prefix $MODEL_PREFIX --model_dim $MODEL_DIM >> $OUT 2>&1
	python comb_stoch_run_cqr.py --qr_training_flag False --comb_calibr_flag False --comb_idx $i --model_prefix $MODEL_PREFIX --model_dim $MODEL_DIM >> $OUT 2>&1
done

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
OUT="${LOGS}${DATE}_${TIME}_out.txt"

MODEL_PREFIX="GRN" 
MODEL_DIM=4
for i in 0 1 2 3 4 5 6
do
	echo $MODEL_PREFIX$MODEL_DIM i: $i
	python comb_stoch_run_cqr.py --qr_training_flag False --comb_calibr_flag True --comb_idx $i --model_prefix $MODEL_PREFIX --model_dim $MODEL_DIM >> $OUT 2>&1
	python comb_stoch_run_cqr.py --qr_training_flag False --comb_calibr_flag False --comb_idx $i --model_prefix $MODEL_PREFIX --model_dim $MODEL_DIM >> $OUT 2>&1
done

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
OUT="${LOGS}${DATE}_${TIME}_out.txt"

MODEL_PREFIX="MRH" 
MODEL_DIM=4
for i in 0 1 2 3 4 5 6 
do
	echo $MODEL_PREFIX$MODEL_DIM i: $i
	python comb_stoch_run_cqr.py --qr_training_flag False --comb_calibr_flag True --comb_idx $i --model_prefix $MODEL_PREFIX --model_dim $MODEL_DIM >> $OUT 2>&1
	python comb_stoch_run_cqr.py --qr_training_flag False --comb_calibr_flag False --comb_idx $i --model_prefix $MODEL_PREFIX --model_dim $MODEL_DIM >> $OUT 2>&1
done

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
OUT="${LOGS}${DATE}_${TIME}_out.txt"

MODEL_PREFIX="GRN" 
MODEL_DIM=6
for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 
do
	echo $MODEL_PREFIX$MODEL_DIM i: $i
	python comb_stoch_run_cqr.py --qr_training_flag False --comb_calibr_flag True --comb_idx $i --model_prefix $MODEL_PREFIX --model_dim $MODEL_DIM >> $OUT 2>&1
	python comb_stoch_run_cqr.py --qr_training_flag False --comb_calibr_flag False --comb_idx $i --model_prefix $MODEL_PREFIX --model_dim $MODEL_DIM >> $OUT 2>&1
done


DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
OUT="${LOGS}${DATE}_${TIME}_out.txt"

MODEL_PREFIX="MRH" 
MODEL_DIM=8
for i in 0 1 2 3 4 5 6 
do
	echo $MODEL_PREFIX$MODEL_DIM i: $i
	python comb_stoch_run_cqr.py --qr_training_flag False --comb_calibr_flag True --comb_idx $i --model_prefix $MODEL_PREFIX --model_dim $MODEL_DIM >> $OUT 2>&1
	python comb_stoch_run_cqr.py --qr_training_flag False --comb_calibr_flag False --comb_idx $i --model_prefix $MODEL_PREFIX --model_dim $MODEL_DIM >> $OUT 2>&1
done