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
1. Run the main script:
   python main.py
