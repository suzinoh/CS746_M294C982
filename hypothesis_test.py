# ==========================================
# FILE: hypothesis_tests.py
# PURPOSE: Perform formal H0 / H1 hypothesis tests 
#          on the master panel econometric model.
# ==========================================

import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

def run_hypothesis_tests():
    print("==========================================")
    print("  Econometric Hypothesis Testing (H0 vs H1)")
    print("==========================================")
    
    file_path = "data/master_panel_features.csv"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Run main.py / feature_eng.py first.")
        return
        
    df = pd.read_csv(file_path)
    
    target_var = "log_Enrollment"
    features = ["log_Average_Total", "log_NGDPDPC", "PCPIE"]
    available_features = [f for f in features if f in df.columns]
    
    clean_df = df.dropna(subset=[target_var] + available_features).copy()
    
    if clean_df.empty:
        print("Error: Clean dataset is empty.")
        return

    # Fit the Academic-Level Fixed Effects Model
    if "Entity_ID" in clean_df.columns and clean_df["Entity_ID"].nunique() > 1:
        formula = f"{target_var} ~ {' + '.join(available_features)} + C(Entity_ID)"
    else:
        formula = f"{target_var} ~ {' + '.join(available_features)}"
        
    model = smf.ols(formula, data=clean_df).fit()
    
    # ----------------------------------------------------
    # Test 1: Overall Model Significance (Overall F-Test)
    # ----------------------------------------------------
    print("\n------------------------------------------")
    print(" 1. OVERALL MODEL SIGNIFICANCE (F-Test)")
    print("------------------------------------------")
    print("H0: All independent variables have zero effect (Beta_1 = Beta_2 = ... = 0)")
    print("H1: At least one independent variable has a non-zero effect")
    print(f"F-statistic: {model.fvalue:.4f}")
    print(f"p-value:     {model.f_pvalue:.4e}")
    
    if model.f_pvalue < 0.05:
        print("-> Conclusion: REJECT H0. The overall econometric model is statistically significant.")
    else:
        print("-> Conclusion: Fail to reject H0.")

    # ----------------------------------------------------
    # Test 2: Individual Coefficient Significance (t-Tests)
    # ----------------------------------------------------
    print("\n------------------------------------------")
    print(" 2. INDIVIDUAL COEFFICIENT SIGNIFICANCE (t-Tests)")
    print("------------------------------------------")
    
    params = model.params
    pvalues = model.pvalues
    conf_int = model.conf_int()
    
    for feat in available_features:
        if feat in params:
            coef = params[feat]
            pval = pvalues[feat]
            ci = conf_int.loc[feat]
            
            print(f"\nVariable: {feat}")
            print(f"  H0: Beta_{feat} = 0 (No effect on international student enrollment)")
            print(f"  H1: Beta_{feat} != 0 (Significant effect on enrollment)")
            print(f"  Estimated Coefficient : {coef:.4f}")
            print(f"  p-value (Two-tailed)  : {pval:.4f}")
            print(f"  95% Confidence Interval: [{ci[0]:.4f}, {ci[1]:.4f}]")
            
            if pval < 0.05:
                print(f"  -> Conclusion: REJECT H0 at alpha = 0.05. '{feat}' is statistically significant.")
            else:
                print(f"  -> Conclusion: Fail to reject H0. '{feat}' is not statistically significant.")

    # ----------------------------------------------------
    # Test 3: Directional Economic Theory (One-Tailed Tests)
    # ----------------------------------------------------
    print("\n------------------------------------------")
    print(" 3. DIRECTIONAL THEORY TESTS (One-Tailed)")
    print("------------------------------------------")
    for feat in available_features:
        if feat in params:
            coef = params[feat]
            pval_two = pvalues[feat]
            # Convert two-tailed p-value to one-tailed p-value safely
            pval_one = pval_two / 2 
            
            print(f"\nVariable: {feat} (Observed Effect Direction: {'Positive' if coef > 0 else 'Negative'})")
            print(f"  One-tailed p-value: {pval_one:.4f}")
            if pval_one < 0.05:
                print(f"  -> Conclusion: Statistically significant in this directional path at alpha = 0.05.")
            else:
                print(f"  -> Conclusion: Not statistically significant for the one-tailed directional test.")
                
    print("\n==========================================")
    print(" Hypothesis Testing Execution Completed.  ")
    print("==========================================")

if __name__ == "__main__":
    run_hypothesis_tests()