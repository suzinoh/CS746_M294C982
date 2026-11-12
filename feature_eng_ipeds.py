# ==========================================
# FILE: feature_eng_ipeds.py
# PURPOSE: Extract IPEDS institutional cost features, 
#          generate EDA plots, and execute 80/20 train-test OLS modeling.
# ==========================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def process_ipeds_and_country_metadata(WDICountry_clean, cost1_df, drvcost_df, drvef_df):
    """
    Process and clean unused IPEDS institutional cost tables and World Bank country metadata.
    """
    print("--- Processing IPEDS Institutional Costs & WDI Metadata ---")
    
    features_dict = {}
    
    # 1. Process World Bank Country Metadata safely
    if not WDICountry_clean.empty:
        candidate_cols = ["Country Code", "Short Name", "Region", "Income Group", "IncomeGroup"]
        available_cols = [c for c in candidate_cols if c in WDICountry_clean.columns]
        
        wdi_features = WDICountry_clean[available_cols].copy()
        wdi_features.rename(
            columns={"Country Code": "Country_Code", "Income Group": "IncomeGroup"},
            inplace=True,
        )
        features_dict["wdi_meta"] = wdi_features
        print(f"Processed WDI metadata shape: {wdi_features.shape}")
    
    # 2. Process IPEDS Institutional Cost Datasets
    ipeds_dfs = {"cost1": cost1_df, "drvcost": drvcost_df, "drvef": drvef_df}
    for name, df in ipeds_dfs.items():
        if not df.empty:
            clean_df = df.copy()
            clean_df.columns = [str(c).strip().replace(" ", "_") for c in clean_df.columns]
            features_dict[name] = clean_df
            print(f"Processed IPEDS table '{name}' with shape {clean_df.shape}")
            
    return features_dict


def run_ipeds_eda(df):
    """
    NEW: Generate and save an exploratory data analysis scatter plot 
    illustrating IPEDS tuition vs. total out-of-state institutional costs.
    """
    print("\n--- Generating IPEDS Exploratory Data Analysis Plot ---")
    os.makedirs("figures", exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    if df.empty or "COTSON" not in df.columns or "TUFEYR3" not in df.columns:
        print("Required IPEDS cost columns ('COTSON', 'TUFEYR3') not found for EDA plotting.")
        return
        
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=df, 
        x="TUFEYR3", 
        y="COTSON", 
        alpha=0.6, 
        color="#1f77b4",
        ax=ax
    )
    ax.set_title("U.S. Institutional Tuition vs. Total Out-of-State Cost (IPEDS)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Base Tuition Rate (TUFEYR3)", fontsize=11, labelpad=8)
    ax.set_ylabel("Total Out-of-State Cost of Attendance (COTSON)", fontsize=11, labelpad=8)
    
    ax.xaxis.set_major_formatter("${x:,.0f}")
    ax.yaxis.set_major_formatter("${x:,.0f}")
    
    fig_path = "figures/ipeds_tuition_vs_total_cost.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved IPEDS EDA scatter plot to {fig_path}")
    plt.close(fig)


def run_ipeds_modeling_pipeline(file_path):
    """
    Execute 80/20 train-test split OLS regression modeling on exported IPEDS features.
    """
    print("\n--- Starting IPEDS Institutional Modeling Pipeline (80/20 Split) ---")
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return None
        
    df = pd.read_csv(file_path)
    
    preferred_target = "COTSON"
    preferred_predictors = ["TUFEYR3", "CINDON", "CINSON", "ENRTOT", "FTE"]
    
    if preferred_target in df.columns:
        target_col = preferred_target
        predictor_cols = [c for c in preferred_predictors if c in df.columns]
    else:
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "UNITID"]
        if len(numeric_cols) < 2:
            print("Error: Not enough numeric columns found for regression modeling.")
            return None
        target_col = numeric_cols[0]
        predictor_cols = numeric_cols[1:5]
        
    clean_df = df[[target_col] + predictor_cols].dropna().copy()
    print(f"Clean samples for IPEDS modeling: {clean_df.shape[0]}")
    
    # 80/20 Train-Test Split
    train_df, test_df = train_test_split(clean_df, test_size=0.20, random_state=42)
    print(f"Training set: {train_df.shape[0]} samples | Test set: {test_df.shape[0]} samples")
    
    # Fit OLS Regression Model on Training Set
    formula = f"{target_col} ~ {' + '.join(predictor_cols)}"
    print(f"Fitting model with formula: {formula}")
    model = smf.ols(formula, data=train_df).fit()
    print(model.summary().tables[1])
    
    # Evaluate on Test Set
    preds = model.predict(test_df)
    rmse = np.sqrt(mean_squared_error(test_df[target_col], preds))
    r2 = r2_score(test_df[target_col], preds)
    
    print(f"\nIPEDS Model Test Evaluation -> RMSE: {rmse:.4f} | Test R2: {r2:.4f}")
    print("--- IPEDS Modeling Pipeline Completed Successfully ---")
    return model


def export_and_model_ipeds(features_dict):
    """
    Merge IPEDS cost and enrollment tables, trigger EDA, export, and run modeling.
    """
    os.makedirs("data", exist_ok=True)
    
    if "drvcost" in features_dict and not features_dict["drvcost"].empty:
        merged_ipeds = features_dict["drvcost"].copy()
        if "drvef" in features_dict and "UNITID" in features_dict["drvef"].columns:
            ef_cols = [c for c in ["UNITID", "ENRTOT", "FTE"] if c in features_dict["drvef"].columns]
            merged_ipeds = pd.merge(merged_ipeds, features_dict["drvef"][ef_cols], on="UNITID", how="left")
    elif "cost1" in features_dict and not features_dict["cost1"].empty:
        merged_ipeds = features_dict["cost1"].copy()
    else:
        print("No valid IPEDS tables available to export.")
        return

    output_path = "data/processed_ipeds_features.csv"
    merged_ipeds.to_csv(output_path, index=False)
    print(f"Successfully exported IPEDS features to {output_path} (shape: {merged_ipeds.shape})")
    
    # Generate exploratory plot
    run_ipeds_eda(merged_ipeds)
    
    # Run modeling
    run_ipeds_modeling_pipeline(output_path)


if __name__ == "__main__":
    from get_data import load_data, clean_and_prepare_panel_data
    
    datasets = load_data()
    wdi_c, imf_c, ipeds_c = clean_and_prepare_panel_data(*datasets)
    
    ipeds_features = process_ipeds_and_country_metadata(
        wdi_c, datasets[1], datasets[4], datasets[5]
    )
    export_and_model_ipeds(ipeds_features)