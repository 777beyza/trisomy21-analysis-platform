import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

class FutureProjectionModel:
    def __init__(self, model_path="models/m3_rf_model.pkl", scaler_path="models/m3_scaler.pkl"):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        
    def train(self, data_path="data/disease_risk_dataset.csv"):
        print("Loading Module 3 data...")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path} bulunamadı. Lütfen önce veri üreticiyi çalıştırın.")
            
        df = pd.read_csv(data_path)
        
        # Inputs: Gene Expressions
        X = df[['APP_Expr', 'DYRK1A_Expr', 'RCAN1_Expr']]
        
        # Outputs: Multiple Disease Risks
        y = df[['Alzheimer_Risk', 'Cardiac_Defect_Risk', 'Cognitive_Defect_Risk', 'Leukemia_Risk', 'Thyroid_Risk']]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("Scaling features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Training Multi-Output Random Forest Regressor...")
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model = MultiOutputRegressor(rf)
        self.model.fit(X_train_scaled, y_train)
        
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R2 Score: {r2:.4f}")
        
        # Save models
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        print(f"Module 3 Model and scaler saved to {os.path.dirname(self.model_path)}")
        
    def load_model(self):
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
        else:
            raise FileNotFoundError("Model dosyaları bulunamadı. Lütfen önce 'train' metodunu çalıştırın.")
            
    def predict_future_risks(self, gene_expressions):
        """
        gene_expressions: list or array [APP_Expr, DYRK1A_Expr, RCAN1_Expr]
        Returns: Dictionary of predicted risks
        """
        if self.model is None or self.scaler is None:
            self.load_model()
            
        features_scaled = self.scaler.transform([gene_expressions])
        predictions = self.model.predict(features_scaled)[0]
        
        risk_labels = ['Erken Başlangıçlı Alzheimer', 'Kalp Defektleri', 'Bilişsel Gerileme', 'Lösemi (AMKL)', 'Tiroid Disfonksiyonu']
        risk_dict = dict(zip(risk_labels, predictions))
        
        return risk_dict

if __name__ == "__main__":
    from data_generator import generate_disease_risk_data
    
    # Generate data if it doesn't exist
    if not os.path.exists("data/disease_risk_dataset.csv"):
        generate_disease_risk_data()
        
    # Train the model
    system = FutureProjectionModel()
    system.train()
