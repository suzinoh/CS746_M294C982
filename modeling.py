# ==========================================
# FILE: modeling.py
# PURPOSE: Implement baseline OLS and proposed 
#          Country Fixed Effects panel models.
# ==========================================
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def run_modeling_pipeline():
    print("--- Starting Econometric Modeling Pipeline ---")
    
    # 1. Load Master Feature Set
    file_path = "data/master_panel_features.csv"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Run feature_eng.py first.")
        return None, None
        
    df = pd.read_csv(file_path)
    print(f"Loaded dataset shape: {df.shape}")
    
    # Define core target and desired features dynamically
    target_var = "log_Enrollment"
    desired_features = ["log_Average_Total", "log_NGDPDPC", "PCPIE"]
    
    # Filter features to only include columns that actually exist in the CSV
    available_features = [v for v in desired_features if v in df.columns]
    missing_features = set(desired_features) - set(available_features)
    if missing_features:
        print(f"Warning: The following desired features were missing and skipped: {missing_features}")
        
    if target_var not in df.columns:
        print(f"Error: Target variable '{target_var}' not found in dataset.")
        return None, None
        
    if not available_features:
        print("Error: No valid independent model features available.")
        return None, None

    # Drop rows missing core regression variables
    subset_cols = [target_var] + available_features
    clean_df = df.dropna(subset=subset_cols).copy()
    print(f"Dataset shape after dropping NaNs: {clean_df.shape}")
    
    if clean_df.empty:
        print("Error: Clean dataset is empty. Check your feature engineering pipeline.")
        return None, None

    # Implement an 80/20 Train-Test Split
    train_df, test_df = train_test_split(clean_df, test_size=0.2, random_state=42)
    print(f"Training set shape: {train_df.shape} | Testing set shape: {test_df.shape}")
    
    # Build formula dynamically based on available features
    formula_base = f"{target_var} ~ {' + '.join(available_features)}"

    # 2. Baseline Model: Pooled OLS Regression (Fit on Train)
    print("\n--- Fitting Baseline Model (Pooled OLS) ---")
    baseline_model = smf.ols(formula_base, data=train_df).fit()
    print(baseline_model.summary().tables[1])
    print(f"Baseline Training R-squared: {baseline_model.rsquared:.4f}")
    
    # 3. Proposed Model: Academic-Level Fixed Effects Panel Regression
    print("\n--- Fitting Proposed Model (Academic-Level Fixed Effects) ---")
    if "Entity_ID" in train_df.columns and train_df["Entity_ID"].nunique() > 1:
        # Include Academic Level categorical dummies to absorb tier-specific baseline differences
        fe_formula = f"{formula_base} + C(Entity_ID)"
        fe_model = smf.ols(fe_formula, data=train_df).fit()
        print(f"Proposed Fixed Effects Training R-squared: {fe_model.rsquared:.4f}")
    else:
        print("Entity_ID column not found or single entity; falling back to baseline specification.")
        fe_model = baseline_model
        
    # 4. Out-of-Sample Evaluation Metrics Calculation (Evaluated on Test Data)
    test_predictions = fe_model.predict(test_df)
    test_rmse = np.sqrt(mean_squared_error(test_df[target_var], test_predictions))
    print(f"\nModel Evaluation Metrics (Out-of-Sample Test Set) -> RMSE: {test_rmse:.4f}, Training R2: {fe_model.rsquared:.4f}")
    
    return baseline_model, fe_model