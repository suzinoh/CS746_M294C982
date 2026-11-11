# ==========================================
# FILE: main.py
# AUTHOR: Suzin Oh
# PROJECT: CS746 / CS797 International Student Mobility
# PURPOSE: End-to-end orchestration script for data loading, 
#          cleaning, EDA, feature engineering, and modeling.
# ==========================================

import os
import sys

# Import custom project modules
from get_data import load_data, clean_and_prepare_panel_data
from exploratory_data_analysis import run_eda
from feature_eng import (
    harmonize_country_identifiers,
    build_panel_dataset,
    apply_lags_and_transformations,
    export_master_features
)
from modeling import run_modeling_pipeline


def main():
    print("============================================")
    print("    CS746/CS797 International Student Mobility")
    print("      Macroeconomic Driver Pipeline           ")
    print("============================================\n")

    # ----------------------------------------------------
    # 1. Load Raw & Cleaned Individual Datasets
    # ----------------------------------------------------
    print("[Step 1/5] Loading raw datasets...")
    (
        academic_level_df,
        cost1_df,
        WDICountry_df,
        IMF_df,
        drvcost_df,
        drvef_df,
        tabn330_df,
    ) = load_data()

    # ----------------------------------------------------
    # 2. Post-Load Cleaning & Panel Harmonization
    # ----------------------------------------------------
    print("\n[Step 2/5] Performing post-load cleaning...")
    WDICountry_clean, IMF_clean, ipeds_clean_merged = clean_and_prepare_panel_data(
        academic_level_df,
        cost1_df,
        WDICountry_df,
        IMF_df,
        drvcost_df,
        drvef_df,
        tabn330_df,
    )

    # Summary Verification
    print("\n--- Ingestion & Cleaning Summary ---")
    print(f"1. Academic Level Enrolment Data: {academic_level_df.shape if not academic_level_df.empty else 'Empty'}")
    print(f"2. NCES Digest Tuition Data: {tabn330_df.shape if not tabn330_df.empty else 'Empty'}")
    print(f"3. Cleaned WDI Country Metadata: {WDICountry_clean.shape if not WDICountry_clean.empty else 'Empty'}")
    print(f"4. Cleaned IMF Economic Indicators: {IMF_clean.shape if not IMF_clean.empty else 'Empty'}")

    # ----------------------------------------------------
    # 3. Exploratory Data Analysis (EDA)
    # ----------------------------------------------------
    print("\n[Step 3/5] Running exploratory data analysis and generating plots...")
    run_eda(
        academic_level_df,
        cost1_df,
        WDICountry_df,
        IMF_df,
        drvcost_df,
        drvef_df,
        tabn330_df,
    )

    # ----------------------------------------------------
    # 4. Feature Engineering & Panel Dataset Construction
    # ----------------------------------------------------
    print("\n[Step 4/5] Executing feature engineering and panel harmonization...")
    harmonized_academic = harmonize_country_identifiers(academic_level_df, WDICountry_clean)
    
    master_panel = build_panel_dataset(
        harmonized_academic,
        cost1_df,
        WDICountry_clean,
        IMF_clean,
        drvcost_df,
        drvef_df,
        tabn330_df
    )
    
    transformed_panel = apply_lags_and_transformations(master_panel)
    export_master_features(transformed_panel)

    # ----------------------------------------------------
    # 5. Econometric Modeling & Evaluation
    # ----------------------------------------------------
    print("\n[Step 5/5] Training baseline OLS and Country Fixed Effects models...")
    baseline_model, fe_model = run_modeling_pipeline()

    print("\n============================================")
    print(" Pipeline Execution Finished Successfully!  ")
    print("============================================")


if __name__ == "__main__":
    main()