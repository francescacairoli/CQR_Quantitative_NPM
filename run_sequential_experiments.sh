#!/bin/bash

############
# settings #
############

LOGS="out/logs/"
mkdir -p $LOGS


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
