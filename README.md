# Conditional Generative Model for Inverse Modeling and Uncertainty Quantification

This repository implements a general-purpose conditional generative model, capable of calibrating models, quantifying uncertainty, and making forecasts efficiently. Below are three illustrative examples showcasing its capabilities:


![Conditional Generative Model](ConditionalGenenativeModel.png)

# Key Capabilities:

<ol>
  <li>Replaces traditional inverse modeling methods (MCMC + surrogate models)</li>
  <li>Predicts model input parameters directly from observations after trainingm</li>
  <li>Generates model output predictions efficiently based on new observations</li>
  <li>Delivers ensemble forecasts with comprehensive uncertainty quantification</li>
  <li>High computational efficiency and storage efficiency</li>
</ol>

# Three Examples:

## Example 1: Bimodal Function Calibration
We calibrate input parameter x given y:  `y = x² + θ`, where `θ` represents random perturbations, and quantify the associated uncertainty of input paramter x.

### Instructions
1. In the config.py, modify the following: 
   
   USE_SYNTHETIC_DATA = True
   
2. Run the main script:
   python main.py
3. The results, including parameter posterior samples and predictive uncertainty, will be saved in the Results directory.

## Example 2: Calibrating the ELM Model at the Missouri Ozark AmeriFlux Site
We calibrate the Ecosystem Land Model (ELM) using observational data from the Missouri Ozark AmeriFlux site. The objective is to quantify the uncertainty for eight sensitive parameters, given five observation variables.

### Instructions
1. In the config.py, modify the following: 
   
   USE_SYNTHETIC_DATA = False
   
   X_SAMPLE_PATH = 'Dataset/ELM/InputPara.npy'   # Model parameters (X)
   
   Y_SAMPLE_PATH = 'Dataset/ELM/Observations.npy'     # Observations (Y)
   
3. python main.py
4. The results, including parameter posterior samples and predictive uncertainty, will be saved in the Results directory.

## Example 3: High-Dimensional Input Parameters – Geological Carbon Storage Case
We forecast the full 2D pressure distribution (64x128 grid) for a geological storage application. Observations are from 10 monitoring points at the injection well.

### Instructions
1. Download the dataset:
   Dataset: dP_train_u.pt [Download here](https://drive.google.com/drive/folders/1fZQfMn_vsjKUXAfRV0q_gswtl8JEkVGo)
   Refer to the original article: https://www-sciencedirect-com.ornl.idm.oclc.org/science/article/pii/S0309170822000562#d1e8497
2. Run dimension reduction: python AE_V2.py
   This reduces the forecast variables from the original 64x128 grid to a latent dimension of 20.
3. In the config.py, modify the following: 
   
   USE_SYNTHETIC_DATA = False
   
   X_SAMPLE_PATH = 'Dataset/Co2_Pressure/pre_latent.npy'   # Model parameters (X)
   
   Y_SAMPLE_PATH = 'Dataset/Co2_Pressure/pre_observation.npy'     # Observations (Y)
   
5. Run the main script for forecasting: python main.py
6. Evaluate testing cases: python Evaluation_co2.py
7. Transform the latent dimensions back to the original space:  python Pres_pred.py
8. The prediction and reconstuction results will be saved in the Results directory.


