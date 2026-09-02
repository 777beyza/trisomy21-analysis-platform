import pandas as pd
import os
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

class EarlyDiagnosisModel:
    def __init__(self, model_path="models/rf_model.pkl", scaler_path="models/scaler.pkl"):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        
    def train(self, data_path="data/cfdna_dataset.csv"):
        print("Loading data...")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path} bulunamadı. Lütfen önce veri üreticiyi çalıştırın.")
            
        df = pd.read_csv(data_path)
        X = df.drop('Trisomy_21', axis=1)
        y = df['Trisomy_21']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print("Scaling features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Training Random Forest...")
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        self.model.fit(X_train_scaled, y_train)
        
        y_pred = self.model.predict(X_test_scaled)
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save models
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        print(f"Model and scaler saved to {os.path.dirname(self.model_path)}")
        
    def load_model(self):
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
        else:
            raise FileNotFoundError("Model dosyaları bulunamadı. Lütfen önce 'train' metodunu çalıştırın.")
            
    def predict_risk(self, patient_features):
        """
        patient_features: list veya array [Maternal_Age, Fetal_Fraction, Chr21_Read_Ratio, GC_Content_Bias, Fragment_Size_Ratio]
        Returns: risk_score (0-100), prediction (0 or 1)
        """
        if self.model is None or self.scaler is None:
            self.load_model()
            
        features_scaled = self.scaler.transform([patient_features])
        prob = self.model.predict_proba(features_scaled)[0]
        prediction = self.model.predict(features_scaled)[0]
        
        risk_score = prob[1] * 100 # Percentage of being positive
        return risk_score, prediction

    def get_shap_explanation(self, patient_features):
        """
        Calculates SHAP values for the given patient features.
        Handles different SHAP versions and output shapes.
        """
        import numpy as np
        if self.model is None or self.scaler is None:
            self.load_model()
            
        features_scaled = self.scaler.transform([patient_features])
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(features_scaled)
        
        # Handle different SHAP versions and output shapes
        if isinstance(shap_values, list):
            # List of arrays [class0_shap, class1_shap] -> each (samples, features)
            # or sometimes (samples, features, 2) inside the list
            v = shap_values[1][0]
            if len(v.shape) == 2 and v.shape[1] == 2: # (features, 2)
                v = v[:, 1]
            e = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        else:
            # Single array
            if len(shap_values.shape) == 3: # (samples, features, classes)
                v = shap_values[0, :, 1]
                e = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
            elif len(shap_values.shape) == 2: # (samples, features)
                v = shap_values[0]
                e = explainer.expected_value
            else:
                v = shap_values
                e = explainer.expected_value
                
        return v, e

    def predict_batch(self, df):
        """
        Analyzes a dataframe of multiple patients.
        """
        if self.model is None or self.scaler is None:
            self.load_model()
        
        # Drop target column and output columns if they exist in the uploaded file
        columns_to_drop = ['Trisomy_21', 'Risk Skoru (%)', 'Teşhis']
        X = df.drop(columns=columns_to_drop, errors='ignore')
        
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)[:, 1] * 100
        preds = self.model.predict(X_scaled)
        
        return probs, preds

    def get_pathway_scores(self, risk_score):
        """
        Simulates pathway enrichment scores based on patient risk.
        """
        import numpy as np
        np.random.seed(42)
        pathways = {
            "Alzheimer's Disease Pathway": 0.85,
            "Heart Development": 0.65,
            "Immune Response": 0.55,
            "Oxidative Stress": 0.70,
            "Neural Stem Cell Proliferation": 0.40,
            "Skeletal Muscle Development": 0.30
        }
        
        # Scale scores based on risk
        multiplier = risk_score / 100.0
        scores = {k: min(100, v * multiplier * 100 + np.random.normal(0, 5)) for k, v in pathways.items()}
        return scores

if __name__ == "__main__":
    from data_generator import generate_cfdna_data
    
    # Generate data if it doesn't exist
    if not os.path.exists("data/cfdna_dataset.csv"):
        generate_cfdna_data()
        
    # Train the model
    system = EarlyDiagnosisModel()
    system.train()
