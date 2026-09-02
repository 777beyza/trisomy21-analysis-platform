import pandas as pd
import numpy as np
import os

def generate_cfdna_data(num_samples=500, save_path="data/cfdna_dataset.csv"):
    """
    Generates synthetic cfDNA sequencing features for Down Syndrome (Trisomy 21) prediction.
    Features:
    - Fetal_Fraction: Fetal DNA percentage.
    - Chr21_Read_Ratio: Proportion of reads mapping to Chromosome 21.
    - GC_Content_Bias: GC content variation.
    - Maternal_Age: Age of the mother (known risk factor).
    - Fragment_Size_Ratio: Ratio of short to long DNA fragments.
    """
    np.random.seed(42)
    
    # 0 = Negative, 1 = Positive (Down Syndrome)
    labels = np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2])
    
    maternal_age = []
    fetal_fraction = []
    chr21_read_ratio = []
    gc_bias = []
    fragment_size_ratio = []
    
    for label in labels:
        if label == 1:
            maternal_age.append(np.random.normal(loc=38, scale=5))
            fetal_fraction.append(np.random.normal(loc=0.15, scale=0.05))
            chr21_read_ratio.append(np.random.normal(loc=1.05, scale=0.02))  # Elevated Chr21 ratio
            gc_bias.append(np.random.normal(loc=0.45, scale=0.03))
            fragment_size_ratio.append(np.random.normal(loc=1.2, scale=0.1)) # Slightly different fragment size
        else:
            maternal_age.append(np.random.normal(loc=28, scale=5))
            fetal_fraction.append(np.random.normal(loc=0.10, scale=0.05))
            chr21_read_ratio.append(np.random.normal(loc=1.00, scale=0.01))  # Normal Chr21 ratio
            gc_bias.append(np.random.normal(loc=0.45, scale=0.03))
            fragment_size_ratio.append(np.random.normal(loc=1.0, scale=0.1))
            
    df = pd.DataFrame({
        'Maternal_Age': maternal_age,
        'Fetal_Fraction': fetal_fraction,
        'Chr21_Read_Ratio': chr21_read_ratio,
        'GC_Content_Bias': gc_bias,
        'Fragment_Size_Ratio': fragment_size_ratio,
        'Trisomy_21': labels
    })
    
    # Ensure realistic bounds
    df['Maternal_Age'] = df['Maternal_Age'].clip(18, 50).astype(int)
    df['Fetal_Fraction'] = df['Fetal_Fraction'].clip(0.01, 0.5)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Dataset generated and saved to {save_path}")
    return df

def generate_disease_risk_data(num_samples=500, save_path="data/disease_risk_dataset.csv"):
    """
    Generates synthetic gene expression levels (Log2FC/Counts) and corresponding
    disease risks (Module 3).
    Key genes for Trisomy 21:
    - APP: Linked to Early-Onset Alzheimer's
    - DYRK1A: Linked to cognitive defects / Alzheimer's
    - RCAN1: Linked to cardiac and cognitive issues
    """
    np.random.seed(42)
    
    # Simulate gene expression levels (Log2FC or normalized counts)
    # Assuming positive values mean over-expression
    app_expr = np.random.normal(loc=2.5, scale=1.0, size=num_samples)
    dyrk1a_expr = np.random.normal(loc=1.8, scale=0.8, size=num_samples)
    rcan1_expr = np.random.normal(loc=2.0, scale=0.9, size=num_samples)
    
    # Calculate risks based on expressions + noise
    # Risks scaled between 0 to 100
    alzheimer_risk = (app_expr * 15) + (dyrk1a_expr * 10) + np.random.normal(0, 5, num_samples)
    cardiac_risk = (rcan1_expr * 20) + np.random.normal(0, 5, num_samples)
    cognitive_risk = (dyrk1a_expr * 12) + (rcan1_expr * 12) + np.random.normal(0, 5, num_samples)
    leukemia_risk = np.random.normal(loc=15, scale=5, size=num_samples) # Background risk
    thyroid_risk = np.random.normal(loc=20, scale=8, size=num_samples) # Background risk
    
    df = pd.DataFrame({
        'APP_Expr': app_expr,
        'DYRK1A_Expr': dyrk1a_expr,
        'RCAN1_Expr': rcan1_expr,
        'Alzheimer_Risk': alzheimer_risk.clip(0, 100),
        'Cardiac_Defect_Risk': cardiac_risk.clip(0, 100),
        'Cognitive_Defect_Risk': cognitive_risk.clip(0, 100),
        'Leukemia_Risk': leukemia_risk.clip(0, 100),
        'Thyroid_Risk': thyroid_risk.clip(0, 100)
    })
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Disease risk dataset generated and saved to {save_path}")
    return df

if __name__ == "__main__":
    generate_cfdna_data()
    generate_disease_risk_data()
