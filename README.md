# Conformal Quantitative Predictive Monitoring of STL Requirements for Stochastic Processes

Authors: Cairoli Francesca, Nicola Paoletti and Luca Bortolussi
University of Trieste, Italy
King's Cross College London, UK

Paper: https://arxiv.org/abs/2211.02375

Submitted to HSCC 23

## Setup

Ubuntu version 18.04.06 LTS

Python version 3.7.13

Install virtual environment:
```
cd src/
apt-get install python3.7-venv
python3.7 -m venv venv
pip3.7 install -r requirements.txt
```

## Dataset generation

Run the following command (on per case study)
```
python AutomAnaesthesiaDelivery.py 
python ExpHeatedTank.py
python generate_multiroom_datasets.py --nb_rooms 2
python generate_generegul_datasets.py --nb_genes 2
```

Datasets are stored in the `Datasets/` folder. Datasets can be downloaded at the following link: https://drive.google.com/drive/folders/15Jt3Mecmu3EFu4GqUUXenS_UrNgkDL8P?usp=share_link

## Inference


Run the following command with the details specific of the case study considered
```
python stoch_run_cqr.py --model_prefix 'MRH' --model_dim 2 --property_idx 0 --qr_training_flag True
```
Model prefixes allowed are 'GRN', 'AAD' and 'EHT'. For 'AAD' and 'EHT' set the property_idx to -1.

For combining different monitors (conjunction of properties) run the following command
```
python comb_stoch_run_cqr.py --model_prefix 'MRH' --model_dim 2 --comb_idx 0 --qr_training_flag True --comb_calibr_flag True
```

The `comb_calibr_flag` is set to false is we want to train the CQR for the conjunction and set to true is we want to combine the property-specific prediction intervals.  `comb_idx` enumerates the possible combinations of properties (without repetions) with the order given by flattening the non-zero elements of the upper-triangolar matrix (no diagonal included).

To efficiently reproduce the experiments run the two following bash commands with the proper model specific settings. All monitors are trained sequentially

```
bash single_exec.py
bash comb_exec.py
```
