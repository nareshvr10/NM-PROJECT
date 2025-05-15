# NM-PROJECT
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import pickle
import warnings
warnings.filterwarnings('ignore')

class HealthcareAIPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importances = {}
        self.disease_descriptions = {
            'diabetes': "Diabetes is a chronic condition that affects how your body processes blood sugar.",
            'heart_disease': "Heart disease refers to various conditions that affect your heart's structure and function.",
            'hypertension': "Hypertension is high blood pressure, a condition that can lead to serious health problems.",
            'chronic_kidney': "Chronic kidney disease means your kidneys are damaged and can't filter blood properly.",
            'stroke': "A stroke occurs when blood supply to part of your brain is interrupted or reduced."
        }
        self.prevention_tips = {
            'diabetes': ["Maintain healthy weight", "Exercise regularly", "Eat balanced diet", 
                         "Monitor blood sugar levels", "Reduce sugar intake"],
            'heart_disease': ["Don't smoke", "Control blood pressure", "Manage cholesterol", 
                              "Exercise regularly", "Eat heart-healthy foods"],
            'hypertension': ["Reduce salt intake", "Manage stress", "Limit alcohol", 
                             "Monitor blood pressure", "Maintain healthy weight"],
            'chronic_kidney': ["Control blood sugar", "Monitor blood pressure", "Stay hydrated", 
                               "Avoid NSAIDs", "Get regular checkups"],
            'stroke': ["Control blood pressure", "Manage cholesterol", "Quit smoking", 
                        "Exercise regularly", "Maintain healthy weight"]
        }

    def load_data(self, filepath):
        """Load patient data from CSV file"""
        try:
            data = pd.read_csv(filepath)
            print(f"Data loaded successfully with {len(data)} records.")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def preprocess_data(self, data, target_column):
        """Preprocess the data: handle missing values, scale features, etc."""
        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test, scaler

    def train_model(self, X_train, y_train, disease_type):
        """Train a Random Forest classifier for disease prediction"""
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        self.models[disease_type] = model
        self.feature_importances[disease_type] = dict(zip(
            [f"feature_{i}" for i in range(X_train.shape[1])], 
            model.feature_importances_
        ))
        print(f"Model trained for {disease_type}")
        return model

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print(f"Model Accuracy: {accuracy:.2f}")
        print("Classification Report:")
        print(report)
        return accuracy, report

    def predict_disease(self, patient_data, disease_type):
        """Predict disease probability for a given patient"""
        if disease_type not in self.models:
            raise ValueError(f"No model trained for {disease_type}")
            
        model = self.models[disease_type]
        scaler = self.scalers.get(disease_type, None)
        
        if scaler:
            patient_data = scaler.transform([patient_data])
        
        probability = model.predict_proba(patient_data)[0][1]
        prediction = model.predict(patient_data)[0]
        
        return {
            'disease': disease_type,
            'prediction': bool(prediction),
            'probability': float(probability),
            'description': self.disease_descriptions.get(disease_type, "No description available"),
            'prevention_tips': self.prevention_tips.get(disease_type, [])
        }

    def save_model(self, model, filename):
        """Save trained model to file"""
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
        print(f"Model saved to {filename}")

    def load_saved_model(self, filename):
        """Load saved model from file"""
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        print(f"Model loaded from {filename}")
        return model

    def generate_health_report(self, predictions):
        """Generate comprehensive health report based on predictions"""
        report = {
            'high_risk_diseases': [],
            'moderate_risk_diseases': [],
            'low_risk_diseases': [],
            'recommendations': [],
            'overall_health_status': 'Good'
        }
        
        for pred in predictions:
            if pred['probability'] > 0.7:
                report['high_risk_diseases'].append(pred)
                report['overall_health_status'] = 'Critical'
            elif pred['probability'] > 0.4:
                report['moderate_risk_diseases'].append(pred)
                if report['overall_health_status'] != 'Critical':
                    report['overall_health_status'] = 'Needs Attention'
            else:
                report['low_risk_diseases'].append(pred)
        
        # Generate recommendations
        for disease in report['high_risk_diseases']:
            report['recommendations'].append({
                'action': 'Immediate consultation needed',
                'disease': disease['disease'],
                'probability': disease['probability']
            })
        
        for disease in report['moderate_risk_diseases']:
            report['recommendations'].append({
                'action': 'Schedule a checkup',
                'disease': disease['disease'],
                'probability': disease['probability']
            })
        
        return report

def main():
    # Initialize the healthcare AI system
    healthcare_ai = HealthcareAIPredictor()
    
    # Example workflow (in a real system, you would load actual patient data)
    print("=== Healthcare AI Disease Prediction System ===")
    
    # For demonstration, we'll create synthetic data
    print("\nGenerating synthetic patient data for demonstration...")
    np.random.seed(42)
    num_patients = 1000
    
    # Create synthetic features
    data = {
        'age': np.random.randint(20, 80, size=num_patients),
        'bmi': np.random.uniform(18, 40, size=num_patients),
        'glucose': np.random.uniform(70, 200, size=num_patients),
        'blood_pressure': np.random.uniform(90, 180, size=num_patients),
        'cholesterol': np.random.uniform(150, 300, size=num_patients),
        'diabetes': np.random.choice([0, 1], size=num_patients, p=[0.85, 0.15]),
        'heart_disease': np.random.choice([0, 1], size=num_patients, p=[0.9, 0.1])
    }
    
    df = pd.DataFrame(data)
    
    # Train diabetes prediction model
    print("\nTraining diabetes prediction model...")
    X_train, X_test, y_train, y_test, scaler = healthcare_ai.preprocess_data(df, 'diabetes')
    healthcare_ai.scalers['diabetes'] = scaler
    diabetes_model = healthcare_ai.train_model(X_train, y_train, 'diabetes')
    healthcare_ai.evaluate_model(diabetes_model, X_test, y_test)
    
    # Train heart disease prediction model
    print("\nTraining heart disease prediction model...")
    X_train, X_test, y_train, y_test, scaler = healthcare_ai.preprocess_data(df, 'heart_disease')
    healthcare_ai.scalers['heart_disease'] = scaler
    heart_model = healthcare_ai.train_model(X_train, y_train, 'heart_disease')
    healthcare_ai.evaluate_model(heart_model, X_test, y_test)
    
    # Example prediction for a new patient
    print("\nMaking predictions for a new patient...")
    new_patient = {
        'age': 55,
        'bmi': 32,
        'glucose': 160,
        'blood_pressure': 140,
        'cholesterol': 280
    }
    patient_features = list(new_patient.values())
    
    diabetes_pred = healthcare_ai.predict_disease(patient_features, 'diabetes')
    heart_pred = healthcare_ai.predict_disease(patient_features, 'heart_disease')
    
    print("\nDiabetes Prediction:")
    print(f"At risk: {diabetes_pred['prediction']} (Probability: {diabetes_pred['probability']:.2%})")
    print(f"Description: {diabetes_pred['description']}")
    print("Prevention Tips:")
    for tip in diabetes_pred['prevention_tips']:
        print(f"- {tip}")
    
    print("\nHeart Disease Prediction:")
    print(f"At risk: {heart_pred['prediction']} (Probability: {heart_pred['probability']:.2%})")
    print(f"Description: {heart_pred['description']}")
    print("Prevention Tips:")
    for tip in heart_pred['prevention_tips']:
        print(f"- {tip}")
    
    # Generate comprehensive health report
    print("\nGenerating comprehensive health report...")
    health_report = healthcare_ai.generate_health_report([diabetes_pred, heart_pred])
    print(f"\nOverall Health Status: {health_report['overall_health_status']}")
    
    if health_report['high_risk_diseases']:
        print("\nHigh Risk Diseases:")
        for disease in health_report['high_risk_diseases']:
            print(f"- {disease['disease']} (Probability: {disease['probability']:.2%})")
    
    if health_report['recommendations']:
        print("\nRecommendations:")
        for rec in health_report['recommendations']:
            print(f"- {rec['action']} for {rec['disease']}")

if __name__ == "__main__":
    main()
