# ML4GASTLI — A Neural-Network Surrogate for GASTLI

> A fast deep-learning emulator of the [GASTLI][gastli] gas-giant interior
> model: predict a planet's radius, entropy and thermal state from its bulk
> parameters in milliseconds, instead of running the full physical model.

![python](https://img.shields.io/badge/python-3.10-3776AB?logo=python&logoColor=white)
![tensorflow](https://img.shields.io/badge/Keras-ResNet-D00000?logo=keras&logoColor=white)
![license](https://img.shields.io/badge/license-BSD--3--Clause-green)

**[GASTLI][gastli]** (the *GAS gianT modeL for Interiors*, [Acuña et al.
2024][paper]) computes interior-structure models for gas giants — mass–radius
relations, thermal evolution and composition retrievals — by solving the
coupled interior/atmosphere physics. That accuracy is expensive: large
parameter sweeps and retrievals require many thousands of evaluations.

**ML4GASTLI** is a Deep Residual Neural Network (ResNet) trained on GASTLI
outputs that reproduces its key predictions almost instantly, so it can stand
in for the full model wherever speed matters (e.g. MCMC retrievals, population
synthesis, interactive exploration).

## Attribution

This surrogate was developed at the **Max Planck Institute for Astronomy
(MPIA), Heidelberg**, in collaboration with **[Lorena Acuña][lorena]** (the
author of GASTLI). The underlying physical model, GASTLI, and its training data
are her work; please credit and cite GASTLI accordingly (see
[Citing](#citing)).

## What it predicts

Given six planetary parameters, the model returns three outputs:

- **Inputs:** mass, core mass fraction, envelope metallicity, core water
  fraction, surface temperature, surface pressure
- **Outputs:** planetary radius, specific entropy, thermal parameter (`f_s`)

### Parameter ranges (training domain)

| Parameter     | Range       | Unit         |
| ------------- | ----------- | ------------ |
| `mass_MEarth` | 0.1 – 600   | Earth masses |
| `CMF`         | 0.0 – 1.0   | fraction     |
| `Zenv`        | 0.0 – 1.0   | fraction     |
| `Zwater_core` | 0.0 – 0.5   | fraction     |
| `Tsurf_K`     | 700 – 6000  | Kelvin       |
| `Psurf_bar`   | 1 – 1000    | bar          |

Inputs outside this domain extrapolate and may be unreliable.

## Installation

```bash
conda create -n ml4gastli python=3.10 -y
conda activate ml4gastli
pip install -r requirements.txt
```

The trained model and scalers are included in [`models/`](models)
(`final_model.keras`, `x_scaler.joblib`, `y_scaler.joblib`), so no extra
downloads are needed.

## Usage

Edit the parameters at the top of [`predict.py`](predict.py):

```python
INPUTS = {
    'mass_MEarth': 100.0,
    'CMF': 0.10,
    'Zenv': 0.10,
    'Zwater_core': 0.0,
    'Tsurf_K': 1000.0,
    'Psurf_bar': 100.0,
}
```

Then run:

```bash
python predict.py
```

### Output

The model returns:

- `radius_Rearth` — planetary radius in Earth radii
- `entropy_SI` — specific entropy in J kg⁻¹ K⁻¹
- `f_s_SI` — thermal parameter in J K⁻¹

Internally the network predicts `log10(radius)` and a signed-log transform of
`f_s`; [`gastli_core.py`](gastli_core.py) converts these back to physical
units automatically.

## Repository layout

```
ML4GASTLI/
├── predict.py        # entry point: set INPUTS, run a prediction
├── gastli_core.py    # GastliModel: loads the network + scalers, runs inference
├── models/           # trained ResNet (.keras) + input/output scalers (.joblib)
├── requirements.txt
├── LICENSE
└── README.md
```

## Citing

If you use this surrogate, please cite the underlying GASTLI model:

> Acuña, L., et al. (2024). *GASTLI: An open-source coupled interior–atmosphere
> model to unveil gas giant composition.* Astronomy & Astrophysics, 688, A60.
> [doi:10.1051/0004-6361/202450559][paper]

- GASTLI code: <https://github.com/lorenaacuna/GASTLI>
- GASTLI docs: <https://gastli.readthedocs.io>

## License

Released under the [BSD 3-Clause License](LICENSE), matching GASTLI.
© 2025 Kunal Bhatia and Lorena Acuña, Max Planck Institute for Astronomy (MPIA).

[gastli]: https://github.com/lorenaacuna/GASTLI
[lorena]: https://lorenaacuna.github.io
[paper]: https://doi.org/10.1051/0004-6361/202450559
