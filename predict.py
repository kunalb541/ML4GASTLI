from gastli_core import GastliModel

# ==========================================
#   STEP 1: DEFINE PLANET PARAMETERS
# ==========================================
# Edit these numbers to simulate different planets
INPUTS = {
    'mass_MEarth': 1.0,      # Mass in Earth Masses (0.1 - 600)
    'CMF': 0.33,             # Core Mass Fraction (0.0 - 1.0)
    'Zenv': 0.02,            # Envelope Metallicity (0.0 - 1.0)
    'Zwater_core': 0.0,      # Core Water Fraction (0.0 - 0.5)
    'Tsurf_K': 1500.0,       # Surface Temperature in Kelvin
    'Psurf_bar': 100.0       # Surface Pressure in Bar
}

def main():
    print("\n" + "="*50)
    print("       GASTLI SURROGATE MODEL INFERENCE")
    print("="*50)

    # Load the model
    print("Status: Loading Neural Network...", end=" ")
    try:
        model = GastliModel(model_dir='models')
        model.load()
        print("Success.")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        print("Make sure you have a folder named 'models' with your .h5 and .joblib files!")
        return

    # Run Prediction
    print("Status: Running Inference...")
    results = model.predict(INPUTS)

    # Display Inputs
    print("-" * 50)
    print(f"INPUTS:")
    print(f"  - Mass:           {INPUTS['mass_MEarth']} M_earth")
    print(f"  - Core Mass Frac: {INPUTS['CMF']}")
    print(f"  - Envelope Metal: {INPUTS['Zenv']}")
    print(f"  - Surface Temp:   {INPUTS['Tsurf_K']} K")
    print(f"  - Surface Press:  {INPUTS['Psurf_bar']} bar")

    # Display Outputs
    print("-" * 50)
    print(f"PREDICTED OUTPUTS:")
    print("-" * 50)
    
    # 1. Radius
    print(f"  1. Radius (R_earth) : {results['radius_Rearth']:.4f}")
    
    # 2. Entropy
    print(f"  2. Entropy (S_1000) : {results['entropy_SI']:.4e}  (J kg^-1 K^-1)")
    
    # 3. Thermal Parameter (Transformed from log space)
    print(f"  3. Thermal Param (fs): {results['f_s_SI']:.4e}  (J K^-1)")
    
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
