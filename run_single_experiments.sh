#!/bin/bash

############
# settings #
############

LOGS="out/logs/"
mkdir -p $LOGS


DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
OUT="${LOGS}${DATE}_${TIME}_out.txt"

# AAD
echo "AAD"
python stoch_run_cqr.py --qr_training_flag False --property_idx -1 --model_prefix "AAD" --model_dim 2 >> $OUT 2>&1

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
OUT="${LOGS}${DATE}_${TIME}_out.txt"

# AAD-F
echo "AAD-F"
python stoch_run_cqr.py --qr_training_flag False --property_idx -1 --model_prefix "AADF" --model_dim 2 >> $OUT 2>&1

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
OUT="${LOGS}${DATE}_${TIME}_out.txt"

# EHT
echo "EHT"
python stoch_run_cqr.py --qr_training_flag False --property_idx -1 --model_prefix "EHT" --model_dim 2 >> $OUT 2>&1

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
OUT="${LOGS}${DATE}_${TIME}_out.txt"

MODEL_PREFIX="MRH"
# MRH 2
for i in 0 1
do
	echo $MODEL_PREFIX i: $i
	python stoch_run_cqr.py --qr_training_flag False --property_idx $i --model_prefix $MODEL_PREFIX --model_dim 2 >> $OUT 2>&1
done

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
OUT="${LOGS}${DATE}_${TIME}_out.txt"

# MRH 4
for i in 0 1 2 3
do
	echo $MODEL_PREFIX i: $i
	python stoch_run_cqr.py --qr_training_flag False --property_idx $i --model_prefix $MODEL_PREFIX --model_dim 4 >> $OUT 2>&1
done

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
OUT="${LOGS}${DATE}_${TIME}_out.txt"

# MRH 8
for i in 0 1 2 3 4 5 6 7
do
	echo $MODEL_PREFIX i: $i
	python stoch_run_cqr.py --qr_training_flag False --property_idx $i --model_prefix $MODEL_PREFIX --model_dim 8 >> $OUT 2>&1
done

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
OUT="${LOGS}${DATE}_${TIME}_out.txt"

MODEL_PREFIX="GRN"
# GRN 2
for i in 0 1
do
	echo $MODEL_PREFIX i: $i
	python stoch_run_cqr.py --qr_training_flag False --property_idx $i --model_prefix $MODEL_PREFIX --model_dim 2 >> $OUT 2>&1
done

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
OUT="${LOGS}${DATE}_${TIME}_out.txt"

# GRN 4
for i in 0 1 2 3
do
	echo $MODEL_PREFIX i: $i
	python stoch_run_cqr.py --qr_training_flag False --property_idx $i --model_prefix $MODEL_PREFIX --model_dim 4 >> $OUT 2>&1
done

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
OUT="${LOGS}${DATE}_${TIME}_out.txt"

# GRN 6
for i in 0 1 2 3 4 5 6 7
do
	echo $MODEL_PREFIX i: $i
	python stoch_run_cqr.py --qr_training_flag False --property_idx $i --model_prefix $MODEL_PREFIX --model_dim 6 >> $OUT 2>&1
done