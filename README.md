GASTLI Surrogate Model

A Fast Machine Learning Surrogate for the GASTLI Interior-Atmosphere Code

This repository contains a trained Neural Network that acts as a surrogate for the GASTLI simulation code. It predicts planetary interior properties in milliseconds.

Repository Contents

predict.py: User Script. Edit this file to define your planet parameters and run the model.

gastli_core.py: Core Logic. Handles loading the model and converting units.

requirements.txt: Dependencies. List of libraries needed to run the code.

models/: (Required) You must create this folder and put your trained files in it.

Installation

Clone or download this repository.

Install the required libraries:

pip install -r requirements.txt


Crucial Step: Create a folder named models in this directory and place your trained files inside it:

final_heteroskedastic_model.h5

X_scaler.joblib

Y_scaler.joblib

How to Run

Open predict.py in a text editor.

Edit the INPUTS section with your planet's Mass, Temperature, etc.

Run the script in your terminal:

python predict.py


Parameter Ranges (Valid Inputs)

Mass: 0.1 to 600 Earth Masses

Core Mass Fraction (CMF): 0.0 to 0.99

Envelope Metallicity (Zenv): 0.0 to 1.0

Core Water (Zwater): 0.0 to 0.5

Surface Temperature: 700 to 6000 K

Surface Pressure: 1 to 1000 bar

Outputs

Radius: In Earth Radii.

Entropy: At 1000 bar (J/kg/K).

Thermal Parameter (fs): Integrated mass-temperature profile (J/K).
