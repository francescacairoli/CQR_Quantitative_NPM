# Conformal Quantitative Predictive Monitoring of STL Requirements for Stochastic Processes


**Authors:** Cairoli Francesca, Nicola Paoletti and Luca Bortolussi
University of Trieste, Italy
King's Cross College London, UK

**Paper:** https://arxiv.org/abs/2211.02375

Accepted to HSCC 23


**Abstract**

We consider the problem of predictive monitoring (PM), i.e., predicting at runtime the satisfaction of a desired property from the current system's state. Due to its relevance for runtime safety assurance and online control, PM methods need to be efficient to enable timely interventions against predicted violations, while providing correctness guarantees. 
We introduce quantitative predictive monitoring (QPM), the first PM method to support stochastic processes and rich specifications given in Signal Temporal Logic (STL). Unlike most of the existing PM techniques that predict whether or not some property $$\phi$$ is satisfied, QPM provides a quantitative measure of satisfaction by predicting the quantitative (aka robust) STL semantics of $$\phi$$. QPM derives prediction intervals that are highly efficient to compute and with probabilistic guarantees, in that the intervals cover with arbitrary probability the STL robustness values relative to the stochastic evolution of the system.  To do so, we take a machine-learning approach and leverage recent advances in conformal inference for quantile regression, thereby avoiding expensive Monte-Carlo simulations at runtime to estimate the intervals. 
We also show how our monitors can be combined in a compositional manner to handle composite formulas, without retraining the predictors nor sacrificing the guarantees. 
We demonstrate the effectiveness and scalability of QPM over a benchmark of four discrete-time stochastic processes with varying degrees of complexity. 


**Download** the pre-generated synthetic datasets and pre-trained models from this repository: https://drive.google.com/drive/folders/15Jt3Mecmu3EFu4GqUUXenS_UrNgkDL8P?usp=sharing
The link contains a compressed file `load.zip`. Unzip the file obtaining `Datasets/` folder containing the pre-generated datasets and the`Models/` folder containing the pre-trained models. Import these two folders in the working directory `src/`.

**Experiments**
    Code is inside the `src/` folder.
Code structure
- `Datasets/`
            This folder contains the pre-generated synthetic datasets. 
- `Models/`
            This folder contains the pre-trained models.
    - `MODEL_PREFIX+MODEL_DIM/` `
            This folder contains a subfolder for each tested configuration: `ID_CQR_#CONFIG_ID_Dropout0.1_multiout_opt=_20hidden_500epochs_3quantiles_3layers_alpha0.1_lr0.0005`
        The string `MODEL_PREFIX+MODEL_DIM` uniquely identifies the case study:
        - `MODEL_PREFIX`: model prefixes are ‘AAD’ for Automated Anaesthesia Delivery, ‘AADF’ for AAD with the eventually operator, ‘EHT’  for Heated Tank with Exponential Failures, ‘MRH’ for Multi Room Heating, ‘GRN’ for Gene Regulatory Network.
        - `MODEL_DIM`: model dimension is set to -1 for ‘AAD’, ‘AADF’ and ‘EHT’. For ‘MRH’ it can be set to 2, 4, 8. For ‘GRN’ it can be set to 2, 4, 6. 
        - `CONFIG_ID`: modular case studies (MRH and GRN) have as many properties as the number of properties. The identifier denotes the considered property. For combined properties, it is a concatenation of the identifiers of the two properties.
- `Results/`
            This folder contains the visualization of the results (as the ones shown in Fig. 6 of the paper) and a file `results.txt` that summarizes the numerical performances (the ones listed in the Tables of the paper). The folder ID is the same used to denote the pre-trained model, uploaded in the `Models/` folder.


![pred_interval_errorbar_merged.png](https://paper-attachments.dropboxusercontent.com/s_DA8D097E86304DE5F96E09771849284B05AB48EC7B955DA117AA1A2D276BF503_1675701233858_pred_interval_errorbar_merged.png)

![sequential_evaluation.png](https://paper-attachments.dropboxusercontent.com/s_DA8D097E86304DE5F96E09771849284B05AB48EC7B955DA117AA1A2D276BF503_1675701816682_sequential_evaluation.png)



            Remark: results of combined properties are stored in the directory of the model with the lowest  index, e.g. when you combine property 0 and property 2 (ID #02), results are stored in folder with ID #0.
- `out/tables/` contains the results in tables (`.csv`  format) with the same structure as those presented in the paper.



**Quick start**
1. Use `src/` as working directory
    cd src
    
2. Create a working virtual environment
    - create a virtual environment
    pip install virtualenv
    python3 -m venv qpm_env
    source qpm_env/bin/activate
    
    - install the specified requirements
    pip install -r requirements.txt
    

In case you have any problem with the pcheck library, download it from: https://github.com/simonesilvetti/pcheck and install it (making sure that the `pcheck/` directory is not nested in other directories).

    cd pcheck
    python setup.py install
    cd ..


**Reproduce experiments**


1. As already stated, the first step consists in downloading the code, the pre-generated synthetic datasets and the pre-trained models from https://drive.google.com/drive/folders/18iEYg0iUsVEYAbyx8Q7nQwiJKpr5fPlj?usp=sharing
    and unzip the file
2. The second step is to set up the virtual environment (as stated in Quick Start)
3. To reproduce all the results presented in the paper run the following bash commands with the proper model-specific settings.


    For single properties: 
    bash run_single_experiments.sh
    For combined properties:
    bash run_conjunction_experiments.sh
    For sequential experiments:
    bash run_sequential_experiments.sh


    In order to run pre-trained models we set `--qr_training_flag False`. To re-train all the models from scratch one should simply set the `--qr_training_flag` to `True`.
    
    In train `run_conjunction_experiments.sh` if the`--comb_calibr_flag` is set to `False` the CQR is trained over the combined property. If it is set to `True`, we combine the intervals coming from two properties.

To **reproduce all the experiments** with a single script run:

    bash run_all.sh

Results are stored in the `Results\` folder with the same ID used to identify the pre-trained model.
To better visualize and compare the output results, the `out\tables\` folder stores the model-specifc tables, in a `.csv` format. These tables summarize the results with the same structure presented in the paper (Table 1 to Table 6).

**Run experiments from scratch**
- Dataset generation: run the following command (one per case study)
    python data_generation/AutomAnaesthesiaDelivery.py 
    python data_generation/ExpHeatedTank.py
    python data_generation/generate_multiroom_datasets.py --nb_rooms MDOEL_DIM
    python data_generation/generate_generegul_datasets.py --nb_genes MODEL_DIM

Setting the desired number of points `nb_points` and the desired number of trajectories `nb_trajs_per_state` to simulate from each state.


- Inference:

Run the following command with the details specific of the case study considered

    python stoch_run_cqr.py --model_prefix MODEL_PREFIX --model_dim MODEL_DIM --property_idx CONFIG_ID --qr_training_flag True

`MODEL_PREFIX` allowed are 'GRN', 'AAD' and 'EHT'. For 'AAD' and 'EHT' set the property_idx is the `CONFIG_ID` defined before.

For combining different monitors (conjunction of properties) run the following command

    python comb_stoch_run_cqr.py --model_prefix MDOEL_PREFIX --model_dim MODEL_DIM --comb_idx 0 --qr_training_flag True --comb_calibr_flag True

The `comb_calibr_flag` is set to false is we want to train the CQR for the conjunction and set to true is we want to combine the property-specific prediction intervals. `comb_idx` enumerates the possible combinations of properties (without repetions) with the order given by flattening the non-zero elements of the upper-triangolar matrix (no diagonal included).

For sequential experiments run

    *_sequential_test.py






