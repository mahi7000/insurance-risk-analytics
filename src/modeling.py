import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, log_loss, mean_absolute_error, r2_score

def prepare_modeling_data(df: pd.DataFrame):
    """Preprocesses variables and converts categorical pillars into model feature matrices."""
    # Build our target columns
    df['IsClaimant'] = (df['TotalClaims'] > 0).astype(int)
    
    # Feature columns for risk assessment
    feature_cols = ['Age', 'Gender', 'Province', 'VehicleType', 'AnnualIncome', 'RiskScore', 'Deductible', 'CustomValueEstimate']
    
    # Drop rows with critical null fields to preserve tracking matrix shapes
    clean_df = df.dropna(subset=feature_cols).copy()
    
    # Generate structural design matrices via dummy encoding
    X = pd.get_dummies(clean_df[feature_cols], drop_first=True)
    
    return clean_df, X

def train_stage_1_classifier(X, y):
    """Trains a Random Forest Classifier to predict the likelihood of a claim event."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    preds = model.predict_proba(X_test)[:, 1]
    metrics = {
        "ROC_AUC": float(roc_auc_score(y_test, preds)),
        "Log_Loss": float(log_loss(y_test, preds))
    }
    return model, metrics, X_test, y_test

def train_stage_2_regressor(X, y):
    """Trains a Random Forest Regressor strictly on positive claimants to predict claim severity."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, preds)),
        "R2": float(r2_score(y_test, preds))
    }
    return model, metrics