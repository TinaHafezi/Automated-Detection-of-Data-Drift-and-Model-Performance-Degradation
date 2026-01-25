import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel


def train_pipeline(data_path="telco.csv"):
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    
    # Preprocessing
    df = df.drop('customerID', axis=1)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Handle TotalCharges
    if df['TotalCharges'].dtype == 'object':
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # One-hot encode
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_encoded, y, test_size=0.4, random_state=42, stratify=y
    )
    
    X_val, X_prod, y_val, y_prod = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    

    # Train initial model for feature selection
    print("\n1. Training initial model for feature selection...")
    base_model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42,
        class_weight='balanced',
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10
    )
    base_model.fit(X_train, y_train)
    
    # Select features
    print("\n2. Selecting important features...")
    selector = SelectFromModel(base_model, prefit=True, threshold='median')
    X_train_selected = selector.transform(X_train)
    X_val_selected = selector.transform(X_val)
    X_prod_selected = selector.transform(X_prod)
    
    # Get selected feature names
    selected_features = X_encoded.columns[selector.get_support()]
    print(f"Selected {len(selected_features)} out of {X_encoded.shape[1]} features")
    print(f"Feature reduction: {100*(1 - len(selected_features)/X_encoded.shape[1]):.1f}%")
    
    # Step 3: Train final model on selected features
    print("\n3. Training final model on selected features...")
    final_model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42,
        class_weight='balanced',
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10
    )
    final_model.fit(X_train_selected, y_train)
    
    # Make predictions
    train_pred = final_model.predict(X_train_selected)
    val_pred = final_model.predict(X_val_selected)
    
    
    # Calculate baseline metrics
    baseline_metrics = {
        "accuracy": accuracy_score(y_val, val_pred),
        "f1": f1_score(y_val, val_pred),
    }
    
    # Feature importance from final model
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE (Top 10)")
    print("="*50)
    
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(10).to_string(index=False))
    
    
    # SAVE ALL ARTIFACTS
    # 1. Save the final model
    joblib.dump(final_model, "model.pkl")
    print("model.pkl - Final trained model")
    
    # 2. Save the feature selector
    joblib.dump(selector, "feature_selector.pkl")
    print("feature_selector.pkl - Feature selector object")
    
    # 3. Save selected feature names
    pd.Series(selected_features).to_csv("selected_features.csv", index=False)
    print("selected_features.csv - List of selected features")
    
    # 4. Save reference data (with all features for monitoring)
    ref_df = pd.concat([X_train, y_train], axis=1)
    ref_df.to_csv("reference.csv", index=False)
    print("reference.csv - Training data with all features")
    
    # 5. Save production data (with all features)
    prod_df = pd.concat([X_prod, y_prod], axis=1)
    prod_df.to_csv("current.csv", index=False)
    print("current.csv - Production data with all features")
    
    # 6. Save baseline metrics
    pd.DataFrame([baseline_metrics]).to_csv("baseline_metrics.csv", index=False)
    print("baseline_metrics.csv - Baseline performance metrics")
    
    # 7. Save all feature columns (for consistency checks)
    pd.Series(X_encoded.columns.tolist()).to_csv("all_features.csv", index=False)
    print("all_features.csv - All original feature names")
    
    # 8. Save selected features data for easy inspection
    X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_features)
    pd.concat([X_train_selected_df, y_train.reset_index(drop=True)], axis=1).to_csv(
        "training_selected_features.csv", index=False
    )
    print("training_selected_features.csv - Training data with selected features only")
    
    print("SUMMARY")
    print(f"Validation Accuracy: {baseline_metrics['accuracy']:.4f}")
    print(f"Validation F1-score: {baseline_metrics['f1']:.4f}")
    print(f"Features: {len(selected_features)} selected out of {X_encoded.shape[1]}")


if __name__ == "__main__":
    train_pipeline()