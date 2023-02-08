# Conformal Quantitative Predictive Monitoring of STL Requirements for Stochastic Processes


Authors: Cairoli Francesca, Nicola Paoletti and Luca Bortolussi
University of Trieste, Italy
King's Cross College London, UK

Paper: https://arxiv.org/abs/2211.02375

Accepted to HSCC 23


# Code structure
- `Datasets/`
- `Models/`
    - `MODEL_PREFIX+MODEL_DIM/QR_results/` 
        This folder contains a subfolder for each tested configuration:
                `ID_CQR_#CONFIG_ID_Dropout0.1_multiout_opt=_20hidden_500epochs_3quantiles_3layers_alpha0.1_lr0.0005`
            This folders contains both the pre-trained models but also the visualization of the results (as the ones shown in Fig. 6 of the paper).
            
![pred_interval_errorbar_merged.png](https://paper-attachments.dropboxusercontent.com/s_DA8D097E86304DE5F96E09771849284B05AB48EC7B955DA117AA1A2D276BF503_1675701233858_pred_interval_errorbar_merged.png)

![sequential_evaluation.png](https://paper-attachments.dropboxusercontent.com/s_DA8D097E86304DE5F96E09771849284B05AB48EC7B955DA117AA1A2D276BF503_1675701816682_sequential_evaluation.png)


The string `MODEL_PREFIX+MODEL_DIM` uniquely identifies the case study:

- `MODEL_PREFIX`: model prefixes are ‘AAD’ for Automated Anaesthesia Delivery, ‘AADF’ for AAD with the eventually operator, ‘EHT’ (Heated Tank with Exponential Failures, ‘MRH’ for Multi Room Heating, GRN for Gene Regulatory Network.
- `MODEL_PREFIX`: model dimension is set to -1 for ‘AAD’, ‘AADF’ and ‘EHT’. For ‘MRH’ it can be set to 2, 4, 8. For ‘GRN’ it can be set to 2, 4, 6. 

# Setup


Create a working virtual environment
    - create a virtual environment
    pip install virtualenv
    python3 -m venv qpm_env
    source qpm_env/bin/activate
    
    - install the specified requirements
    pip install -r requirements.txt
    

Install the pcheck library, download it from: https://github.com/simonesilvetti/pcheck and install it (making sure that the `pcheck/` directory is not nested in other directories).

    cd pcheck
    python setup.py install
    cd ..
    
# Reproduce experiments


1. Download the **pre-generated** synthetic **datasets** from https://drive.google.com/drive/folders/18iEYg0iUsVEYAbyx8Q7nQwiJKpr5fPlj?usp=sharing into the `Datasets/` folder (make sure that the `Datasets/` directory is not nested in other directories)
2. Download the **pre-trained models** from https://drive.google.com/drive/folders/14l2MOAmp64tlOrBEKYQbM4zhvdUT99Bm?usp=sharing into the `Models/` folder (make sure that the `Models/` directory is not nested in other directories)
3. To reproduce all the results presented in the paper by run the two following bash commands with the proper model-specific settings.


    For single properties: 
    
     `bash run_single_experiments.py `
    
    For combined properties:
    
    `bash run_conjunction_experiments.py `


    In order to run pre-trained models we set `--qr_training_flag False`. To re-train all the models from scratch one should simple set the `--qr_training_flag` to `True`.
    
    In train `run_conjunction_experiments.sh` if the`--comb_calibr_flag` is set to `False` the CQR is trained over the combined property. If it is set to `True`, we combine the two properties.


# Run experiments from scratch
- **Dataset generation**: run the following command (one per case study)

    `python AutomAnaesthesiaDelivery.py `
    `python ExpHeatedTank.py`
    `python generate_multiroom_datasets.py --nb_rooms MDOEL_DIM`
    `python generate_generegul_datasets.py --nb_genes MODEL_DIM`

Setting the desired number of points `nb_points` and the desired number of trajectories `nb_trajs_per_state` to simulate from each state.


- **Inference**:

Run the following command with the details specific of the case study considered

     `python stoch_run_cqr.py --model_prefix MODEL_PREFIX --model_dim MODEL_DIM --property_idx CONFIG_ID --qr_training_flag True `

`MODEL_PREFIX` allowed are 'GRN', 'AAD' and 'EHT'. For 'AAD' and 'EHT' set the property_idx is the `CONFIG_ID` defined before.

For **combining** different monitors (conjunction of properties) run the following command

     `python comb_stoch_run_cqr.py --model_prefix MDOEL_PREFIX --model_dim MODEL_DIM --comb_idx 0 --qr_training_flag True --comb_calibr_flag True `

The `comb_calibr_flag` is set to false is we want to train the CQR for the conjunction and set to true is we want to combine the property-specific prediction intervals. `comb_idx` enumerates the possible combinations of properties (without repetions) with the order given by flattening the non-zero elements of the upper-triangolar matrix (no diagonal included).

For **sequential** experiments run

    ` *_sequential_test.py `
