# GASTLI Surrogate Model

A neural network surrogate for the GASTLI planetary interiors.

## Overview

This model predicts three outputs given six planetary parameters:
- **Inputs:** mass, core mass fraction, envelope metallicity, core water fraction, surface temperature, surface pressure
- **Outputs:** planetary radius, entropy, thermal parameter (f_s)

The model provides both mean predictions and uncertainty estimates using heteroskedastic neural networks.

## Parameter Ranges

| Parameter | Range | Unit |
|-----------|-------|------|
| mass_MEarth | 0.1 - 600 | Earth masses |
| CMF | 0.0 - 1.0 | fraction |
| Zenv | 0.0 - 1.0 | fraction |
| Zwater_core | 0.0 - 0.5 | fraction |
| Tsurf_K | 700 - 6000| Kelvin |
| Psurf_bar | 1 - 1000 | bar |

## Installation
```bash
#Create a new environment before
pip install -r requirements.txt
```

## Setup

1. Create a `models/` directory in the repository root
2. Place the following trained model files in `models/`:
   - `final_heteroskedastic_model.h5`
   - `X_scaler.joblib`
   - `Y_scaler.joblib`

## Usage

Edit the parameters in `predict.py`:
```python
INPUTS = {
    'mass_MEarth': 1.0,
    'CMF': 0.33,
    'Zenv': 0.02,
    'Zwater_core': 0.0,
    'Tsurf_K': 1500.0,
    'Psurf_bar': 100.0
}
```

Run the prediction:
```bash
python predict.py
```

## Output

The model returns:
- `radius_Rearth`: Planetary radius in Earth radii
- `entropy_SI`: Specific entropy in J kg⁻¹ K⁻¹
- `f_s_SI`: Thermal parameter in J K⁻¹

## Files

- `predict.py` - Main script for running predictions
- `gastli_core.py` - Model loading and prediction logic
- `t.py` - Training script (reference only)
- `requirements.txt` - Python dependencies

## Notes

- The model uses a signed logarithmic transformation for f_s internally and converts back to physical values automatically
- Predictions include uncertainty estimates (variance/confidence intervals) which are available in the full model output
- Input values outside the training range may produce unreliable predictions