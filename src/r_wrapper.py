import os
import subprocess
import pandas as pd
import numpy as np

def generate_mock_rna_data(filepath, risk_score):
    """
    Generates a mock RNA-Seq count matrix and saves it to a CSV.
    If risk_score is high, it simulates overexpression of Trisomy 21 genes in the patient group.
    """
    np.random.seed(42)
    genes = [f'Gene_{i}' for i in range(1, 1001)]
    
    # Trisomy 21 associated genes
    critical_genes = ['APP', 'SOD1', 'DYRK1A', 'RCAN1', 'DSCR1', 'COL6A1']
    genes[:len(critical_genes)] = critical_genes
    
    # Generate random counts for 3 controls and 3 patients
    # Normal distribution centered around 100 counts
    data = {
        'Gene': genes,
        'Control_1': np.random.normal(100, 20, 1000).clip(10).astype(int),
        'Control_2': np.random.normal(100, 20, 1000).clip(10).astype(int),
        'Control_3': np.random.normal(100, 20, 1000).clip(10).astype(int),
        'Patient_1': np.random.normal(100, 20, 1000).clip(10).astype(int),
        'Patient_2': np.random.normal(100, 20, 1000).clip(10).astype(int),
        'Patient_3': np.random.normal(100, 20, 1000).clip(10).astype(int),
    }
    
    df = pd.DataFrame(data)
    
    # If high risk, heavily upregulate the critical genes in patients
    if risk_score > 40:
        multiplier = (risk_score / 20.0) # e.g. 80 risk = 4x multiplier
        for i, gene in enumerate(critical_genes):
            df.loc[i, 'Patient_1'] = int(df.loc[i, 'Patient_1'] * multiplier * np.random.uniform(0.8, 1.2))
            df.loc[i, 'Patient_2'] = int(df.loc[i, 'Patient_2'] * multiplier * np.random.uniform(0.8, 1.2))
            df.loc[i, 'Patient_3'] = int(df.loc[i, 'Patient_3'] * multiplier * np.random.uniform(0.8, 1.2))
            
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    return filepath

def run_r_analysis(risk_score):
    """
    Generates input data, calls the R script via subprocess, and returns the results DataFrame.
    """
    input_csv = os.path.abspath("data/rna_input.csv")
    output_csv = os.path.abspath("data/rna_results.csv")
    r_script = os.path.abspath("src/deseq_analysis.R")
    
    # 1. Prepare Data
    generate_mock_rna_data(input_csv, risk_score)
    
    # 2. Path to Rscript (Assuming standard Windows installation)
    rscript_path = r"C:\Program Files\R\R-4.6.1\bin\Rscript.exe"
    
    # 3. Call R
    try:
        result = subprocess.run(
            [rscript_path, r_script, input_csv, output_csv],
            capture_output=True,
            text=True,
            check=True
        )
        print("R Script Output:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        print("R Script Error:\n", e.stderr)
        raise Exception(f"R Analiz Motoru Hatası: {e.stderr}")
    except FileNotFoundError:
        raise Exception("Rscript bulunamadı. R'ın C:\\Program Files\\R\\R-4.6.1 yolunda kurulu olduğundan emin olun.")
        
    # 4. Read and return results
    if os.path.exists(output_csv):
        results_df = pd.read_csv(output_csv)
        # Rename column to match existing plotly logic in app.py if needed
        if '-Log10_P' in results_df.columns:
            results_df.rename(columns={'-Log10_P': '-Log10(P-value)'}, inplace=True)
        return results_df
    else:
        raise Exception("R script çalıştı fakat çıktı dosyası oluşturulamadı.")
