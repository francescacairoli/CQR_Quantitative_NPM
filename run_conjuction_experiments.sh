#!/bin/bash

############
# settings #
############

LOGS="out/logs/"
mkdir -p $LOGS



echo "# EXPERIMENTS WITH CONJUNCTION OF PROPERTIES #"

MODEL_PREFIX="GRN" 
MODEL_DIM=2

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
OUT="${LOGS}Conj_${MODEL_PREFIX}${MODEL_DIM}_${DATE}_${TIME}_out.txt"

for i in 0
do
	echo $MODEL_PREFIX$MODEL_DIM i: $i
	python comb_stoch_run_cqr.py --qr_training_flag False --comb_calibr_flag True --comb_idx $i --model_prefix $MODEL_PREFIX --model_dim $MODEL_DIM >> $OUT 2>&1
	python comb_stoch_run_cqr.py --qr_training_flag False --comb_calibr_flag False --comb_idx $i --model_prefix $MODEL_PREFIX --model_dim $MODEL_DIM >> $OUT 2>&1
done


MODEL_PREFIX="MRH" 
MODEL_DIM=2

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
OUT="${LOGS}Conj_${MODEL_PREFIX}${MODEL_DIM}_${DATE}_${TIME}_out.txt"

for i in 0
do
	echo $MODEL_PREFIX$MODEL_DIM i: $i
	python comb_stoch_run_cqr.py --qr_training_flag False --comb_calibr_flag True --comb_idx $i --model_prefix $MODEL_PREFIX --model_dim $MODEL_DIM >> $OUT 2>&1
	python comb_stoch_run_cqr.py --qr_training_flag False --comb_calibr_flag False --comb_idx $i --model_prefix $MODEL_PREFIX --model_dim $MODEL_DIM >> $OUT 2>&1
done


MODEL_PREFIX="GRN" 
MODEL_DIM=4

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
OUT="${LOGS}Conj_${MODEL_PREFIX}${MODEL_DIM}_${DATE}_${TIME}_out.txt"

for i in 0 1 2 3 4 5
do
	echo $MODEL_PREFIX$MODEL_DIM i: $i
	python comb_stoch_run_cqr.py --qr_training_flag False --comb_calibr_flag True --comb_idx $i --model_prefix $MODEL_PREFIX --model_dim $MODEL_DIM >> $OUT 2>&1
	python comb_stoch_run_cqr.py --qr_training_flag False --comb_calibr_flag False --comb_idx $i --model_prefix $MODEL_PREFIX --model_dim $MODEL_DIM >> $OUT 2>&1
done


MODEL_PREFIX="MRH" 
MODEL_DIM=4

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
OUT="${LOGS}Conj_${MODEL_PREFIX}${MODEL_DIM}_${DATE}_${TIME}_out.txt"

for i in 0 1 2 3 4 5 
do
	echo $MODEL_PREFIX$MODEL_DIM i: $i
	python comb_stoch_run_cqr.py --qr_training_flag False --comb_calibr_flag True --comb_idx $i --model_prefix $MODEL_PREFIX --model_dim $MODEL_DIM >> $OUT 2>&1
	python comb_stoch_run_cqr.py --qr_training_flag False --comb_calibr_flag False --comb_idx $i --model_prefix $MODEL_PREFIX --model_dim $MODEL_DIM >> $OUT 2>&1
done


MODEL_PREFIX="GRN" 
MODEL_DIM=6

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
OUT="${LOGS}Conj_${MODEL_PREFIX}${MODEL_DIM}_${DATE}_${TIME}_out.txt"

for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 
do
	echo $MODEL_PREFIX$MODEL_DIM i: $i
	python comb_stoch_run_cqr.py --qr_training_flag False --comb_calibr_flag True --comb_idx $i --model_prefix $MODEL_PREFIX --model_dim $MODEL_DIM >> $OUT 2>&1
	python comb_stoch_run_cqr.py --qr_training_flag False --comb_calibr_flag False --comb_idx $i --model_prefix $MODEL_PREFIX --model_dim $MODEL_DIM >> $OUT 2>&1
done


MODEL_PREFIX="MRH" 
MODEL_DIM=8

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
OUT="${LOGS}Conj_${MODEL_PREFIX}${MODEL_DIM}_${DATE}_${TIME}_out.txt"

for i in 0 1 2 3 4 5 6 
do
	echo $MODEL_PREFIX$MODEL_DIM i: $i
	python comb_stoch_run_cqr.py --qr_training_flag False --comb_calibr_flag True --comb_idx $i --model_prefix $MODEL_PREFIX --model_dim $MODEL_DIM >> $OUT 2>&1
	python comb_stoch_run_cqr.py --qr_training_flag False --comb_calibr_flag False --comb_idx $i --model_prefix $MODEL_PREFIX --model_dim $MODEL_DIM >> $OUT 2>&1
done
