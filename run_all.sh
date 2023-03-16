#!/bin/bash

############
# settings #
############

LOGS="out/logs/"
mkdir -p $LOGS



###############################
# SINGLE PROPERTY EXPERIMENTS #
###############################

echo "# SINGLE PROPERTY EXPERIMENTS #"


DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
OUT="${LOGS}AAD_${DATE}_${TIME}_out.txt"


# AAD
echo "AAD"
python stoch_run_cqr.py --qr_training_flag False --property_idx -1 --model_prefix "AAD" --model_dim 2 >> $OUT 2>&1

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
OUT="${LOGS}AAD-F_${DATE}_${TIME}_out.txt"

# AAD-F
echo "AAD-F"
python stoch_run_cqr.py --qr_training_flag False --property_idx -1 --model_prefix "AADF" --model_dim 2 >> $OUT 2>&1

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
OUT="${LOGS}EHT_${DATE}_${TIME}_out.txt"

# EHT
echo "EHT"
python stoch_run_cqr.py --qr_training_flag False --property_idx -1 --model_prefix "EHT" --model_dim 2 >> $OUT 2>&1

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
OUT="${LOGS}MRH2_${DATE}_${TIME}_out.txt"

MODEL_PREFIX="MRH"
# MRH 2
for i in 0 1
do
	echo $MODEL_PREFIX i: $i
	python stoch_run_cqr.py --qr_training_flag False --property_idx $i --model_prefix $MODEL_PREFIX --model_dim 2 >> $OUT 2>&1
done

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
OUT="${LOGS}MRH4_${DATE}_${TIME}_out.txt"

# MRH 4
for i in 0 1 2 3
do
	echo $MODEL_PREFIX i: $i
	python stoch_run_cqr.py --qr_training_flag False --property_idx $i --model_prefix $MODEL_PREFIX --model_dim 4 >> $OUT 2>&1
done

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
OUT="${LOGS}MRH8_${DATE}_${TIME}_out.txt"

# MRH 8
for i in 0 1 2 3 4 5 6 7
do
	echo $MODEL_PREFIX i: $i
	python stoch_run_cqr.py --qr_training_flag False --property_idx $i --model_prefix $MODEL_PREFIX --model_dim 8 >> $OUT 2>&1
done

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
OUT="${LOGS}GRN2_${DATE}_${TIME}_out.txt"

MODEL_PREFIX="GRN"
# GRN 2
for i in 0 1
do
	echo $MODEL_PREFIX i: $i
	python stoch_run_cqr.py --qr_training_flag False --property_idx $i --model_prefix $MODEL_PREFIX --model_dim 2 >> $OUT 2>&1
done

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
OUT="${LOGS}GRN4_${DATE}_${TIME}_out.txt"

# GRN 4
for i in 0 1 2 3
do
	echo $MODEL_PREFIX i: $i
	python stoch_run_cqr.py --qr_training_flag False --property_idx $i --model_prefix $MODEL_PREFIX --model_dim 4 >> $OUT 2>&1
done

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
OUT="${LOGS}GRN6_${DATE}_${TIME}_out.txt"

# GRN 6
for i in 0 1 2 3 4 5
do
	echo $MODEL_PREFIX i: $i
	python stoch_run_cqr.py --qr_training_flag False --property_idx $i --model_prefix $MODEL_PREFIX --model_dim 6 >> $OUT 2>&1
done


##############################################
# EXPERIMENTS WITH CONJUNCTION OF PROPERTIES #
##############################################
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






###########################
# SEQUENTIAL EXPERIMENTS #
###########################
echo "# SEQUENTIAL EXPERIMENTS #"

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
OUT="${LOGS}Seq_${DATE}_${TIME}_out.txt"

echo "Sequential AAD"
python aad_sequential_test.py --prop_str 'G' --seed 0 >> $OUT 2>&1
python aad_sequential_test.py --prop_str 'G' --seed 2 >> $OUT 2>&1

echo "Sequential AAD-F"
python aad_sequential_test.py --prop_str 'F' --seed 0 >> $OUT 2>&1

echo "Sequential MRH"
python mrh_sequential_test.py --seed 0 >> $OUT 2>&1

echo "Sequential GRN"
python grn_sequential_test.py --seed 0 >> $OUT 2>&1
