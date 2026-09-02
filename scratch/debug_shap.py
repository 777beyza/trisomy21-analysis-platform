import sys
import os
sys.path.append(os.getcwd())
from src.module1_diagnosis import EarlyDiagnosisModel
import numpy as np

model = EarlyDiagnosisModel()
try:
    model.load_model()
    # Dummy features: [maternal_age, fetal_fraction, chr21_ratio, gc_bias, fragment_ratio]
    test_features = [30, 0.1, 1.0, 0.45, 1.0]
    shap_vals, base_val = model.get_shap_explanation(test_features)
    print(f"SHAP vals type: {type(shap_vals)}")
    print(f"SHAP vals shape: {getattr(shap_vals, 'shape', 'N/A')}")
    print(f"SHAP vals content: {shap_vals}")
    
    import shap
    import matplotlib.pyplot as plt
    feature_names = ["Age", "FF", "Chr21", "GC", "Frag"]
    print("Attempting shap.bar_plot...")
    shap.bar_plot(shap_vals, feature_names=feature_names, show=False)
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
