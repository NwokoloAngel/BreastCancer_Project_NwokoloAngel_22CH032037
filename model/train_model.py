import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os

def main():
    print("--- Breast Cancer Prediction Model Training ---")

    # 1. Load Data
    print("Loading breast cancer dataset...")
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # 2. Feature Selection
    # Requirements: radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean
    # internal names: 'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'
    selected_features = [
        'mean radius', 
        'mean texture', 
        'mean perimeter', 
        'mean area', 
        'mean smoothness'
    ]
    
    print(f"Selecting features: {selected_features}")
    
    X = df[selected_features]
    y = df['target'] # 0 = malignant, 1 = benign (Note: sklearn standard)
    
    # Note: The user prompt says "diagnosis (target variable: Benign / Malignant)"
    # In sklearn breast_cancer:
    # 0 = malignant
    # 1 = benign
    
    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Pipeline (Scaling + Model)
    # Using SVM as requested (one of the options)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='linear', random_state=42))
    ])

    # 5. Train Model
    print("Training Support Vector Classifier (SVC)...")
    pipeline.fit(X_train, y_train)

    # 6. Evaluate
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))
    
    # 7. Save Model
    # Ensure directory exists
    model_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(model_dir, 'breast_cancer_model.pkl')
    
    print(f"Saving model pipeline to {model_path}...")
    joblib.dump(pipeline, model_path)
    print("Model saved successfully.")

if __name__ == "__main__":
    main()
