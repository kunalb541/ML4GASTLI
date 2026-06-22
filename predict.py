from gastli_core import GastliModel

# ==========================================
#   STEP 1: DEFINE PLANET PARAMETERS
# ==========================================
INPUTS = {
    'mass_MEarth': 100.0,    # Mass in Earth Masses (0.1 - 600)
    'CMF': 0.10,             # Core Mass Fraction (0.0 - 1.0)
    'Zenv': 0.10,            # Envelope Metallicity (0.0 - 1.0)
    'Zwater_core': 0.0,      # Core Water Fraction (0.0 - 0.5)
    'Tsurf_K': 1000.0,       # Surface Temperature in Kelvin (700 - 6000)
    'Psurf_bar': 100.0       # Surface Pressure in Bar (1 - 1000)
}

def main():
    print("\n" + "="*50)
    print("       GASTLI SURROGATE MODEL INFERENCE")
    print("="*50)

    # Load the model
    print("Status: Loading ResNet Model...", end=" ")
    try:
        model = GastliModel(model_dir='models')
        model.load()
        print("Success.")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        print("Ensure 'models/' contains: final_model.keras, x_scaler.joblib, y_scaler.joblib")
        return

    # Run Prediction
    print("Status: Running Inference...")
    try:
        results = model.predict(INPUTS)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Prediction Error: {e}")
        return

    # Display Inputs
    print("-" * 50)
    print(f"INPUTS:")
    for k, v in INPUTS.items():
        print(f"  - {k:<15}: {v}")

    # Display Outputs
    print("-" * 50)
    print(f"PREDICTED OUTPUTS:")
    print("-" * 50)
    print(f"  1. Radius        : {results['radius_Rearth']:.4f} R_earth")
    print(f"  2. Entropy       : {results['entropy_SI']:.4e} J kg^-1 K^-1")
    print(f"  3. Thermal (fs)  : {results['f_s_SI']:.4e} J K^-1")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
