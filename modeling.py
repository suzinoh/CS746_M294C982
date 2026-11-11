# ==========================================
# FILE: modeling.py
# PURPOSE: Implement baseline OLS and proposed 
#          Country Fixed Effects panel models.
# ==========================================

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error, r2_score

def run_modeling_pipeline():
    print("--- Starting Econometric Modeling Pipeline ---")
    
    # 1. Load Master Feature Set
    file_path = "data/master_panel_features.csv"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Run feature_eng.py first.")
        return
        
    df = pd.read_csv(file_path)
    
    # Drop rows missing core regression variables
    model_vars = ["log_Enrollment", "log_Average_Total", "log_NGDPDPC", "PCPIE"]
    clean_df = df.dropna(subset=model_vars).copy()
    
    print(f"Dataset shape for modeling: {clean_df.shape}")
    
    # 2. Baseline Model: Pooled OLS Regression
    print("\n--- Fitting Baseline Model (Pooled OLS) ---")
    baseline_formula = "log_Enrollment ~ log_Average_Total + log_NGDPDPC + PCPIE"
    baseline_model = smf.ols(baseline_formula, data=clean_df).fit()
    print(baseline_model.summary().tables[1])
    print(f"Baseline R-squared: {baseline_model.rsquared:.4f}")
    
    # 3. Proposed Model: Country Fixed Effects Panel Regression
    print("\n--- Fitting Proposed Model (Country Fixed Effects) ---")
    if "Country_Code" in clean_df.columns and clean_df["Country_Code"].nunique() > 1:
        # Include Country categorical dummies to absorb unobserved heterogeneity
        fe_formula = "log_Enrollment ~ log_Average_Total + log_NGDPDPC + PCPIE + C(Country_Code)"
        fe_model = smf.ols(fe_formula, data=clean_df).fit()
        print(f"Proposed Fixed Effects R-squared: {fe_model.rsquared:.4f}")
    else:
        print("Country_Code column not found or single entity; falling back to global specification.")
        fe_model = baseline_model
        
    # 4. Evaluation Metrics Calculation
    predictions = fe_model.predict(clean_df)
    rmse = np.sqrt(mean_squared_error(clean_df["log_Enrollment"], predictions))
    print(f"\nModel Evaluation Metrics -> RMSE: {rmse:.4f}, R2: {fe_model.rsquared:.4f}")
    
    return baseline_model, fe_model
