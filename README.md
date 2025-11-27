GASTLI Surrogate Model

A Fast Machine Learning Surrogate for the GASTLI Interior-Atmosphere Code

📖 Overview

This repository contains a trained Heteroskedastic Neural Network that serves as a surrogate model for the GASTLI (General Atmospheric Structure and Thermal evolution of Large Interiors) simulation code.

Calculating planetary interiors usually requires computationally expensive physical simulations (taking seconds to hours). This ML model approximates those physics to predict key planetary properties in milliseconds, enabling rapid Bayesian inference (MCMC) and atmospheric retrievals.

📂 Repository Structure

File

Description

predict.py

Main User Script. Edit inputs here to run the model.

gastli_core.py

Backend Logic. Handles model loading, scaling, and physical unit conversions.

requirements.txt

List of Python dependencies.

models/

(Required) Folder containing the trained neural network and scalers.

⚙️ Installation

1. Prerequisite

Ensure you have Python 3.9 or newer installed.

2. Setup

Clone the repository and install the required libraries:

git clone [https://github.com/YOUR_USERNAME/gastli-surrogate-model.git](https://github.com/YOUR_USERNAME/gastli-surrogate-model.git)
cd gastli-surrogate-model
pip install -r requirements.txt


3. Model Artifacts (Crucial Step)

You must create a directory named models in the root folder and populate it with the trained artifacts. These files are too large for standard git and should be provided separately or via Git LFS:

models/final_heteroskedastic_model.h5

models/X_scaler.joblib

models/Y_scaler.joblib

🚀 Usage

Open predict.py in your code editor.

Modify the INPUTS dictionary with your planet parameters:

INPUTS = {
    'mass_MEarth': 5.5,      # Mass (Earth Masses)
    'CMF': 0.33,             # Core Mass Fraction
    'Zenv': 0.02,            # Envelope Metallicity
    'Zwater_core': 0.0,      # Core Water Fraction
    'Tsurf_K': 1500.0,       # Surface Temperature (K)
    'Psurf_bar': 100.0       # Surface Pressure (bar)
}


Run the inference script:

python predict.py


📊 Model Parameters

Inputs

The model is trained on a specific grid of parameters. For best accuracy, keep inputs within these ranges:

Parameter

Symbol

Unit

Valid Range

Description

Mass

$M_p$

$M_\oplus$

0.1 - 600

Planet mass (Earth masses)

CMF

-

-

0.0 - 0.99

Core Mass Fraction

Metallicity

$Z_{env}$

-

0.0 - 1.0

Envelope metal mass fraction

Water (Core)

$Z_{water}$

-

0.0 - 0.5

Water mass fraction in core

Surface Temp

$T_{surf}$

K

700 - 6000

Temperature at boundary

Surface Press

$P_{surf}$

bar

1 - 1000

Pressure at boundary

Outputs

The model predicts the following internal properties:

Radius ($R_p$): The planetary radius in Earth radii ($R_\oplus$).

Entropy ($S_{1000}$): Specific entropy at 1000 bar ($J \cdot kg^{-1} \cdot K^{-1}$).

Thermal Parameter ($f_s$): The integrated mass-temperature profile ($J \cdot K^{-1}$).

Note: The raw model predicts a signed log transformation of $f_s$. The code automatically converts this back to physical units.
