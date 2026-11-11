# ==========================================
# FILE: feature_eng.py
# PURPOSE: Panel data merging, lagging, and 
#          log-log transformations for economic models.
# ==========================================

import os
import numpy as np
import pandas as pd


def harmonize_country_identifiers(academic_level_df, WDICountry_clean):
    """
    Adapted for Academic-Level Panel:
    - Uses Academic_Level as the panel entity unit since data is structured globally by level.
    """
    print("--- Harmonizing Academic Level Entities ---")
    
    if "Academic_Level" in academic_level_df.columns:
        # Clean string spaces and set as our entity identifier
        academic_level_df["Entity_ID"] = academic_level_df["Academic_Level"].astype(str).str.strip()
    else:
        academic_level_df["Entity_ID"] = "GLOBAL_TOTAL"
        
    return academic_level_df


def build_panel_dataset(academic_level_df, cost1_df, WDICountry_clean, IMF_clean, drvcost_df, drvef_df, tabn330_df):
    """
    TODO: 2. Construct Master Panel Dataset
    - Merge IIE Open Doors enrollment data, IMF macroeconomic indicators (GDP per capita, CPI), 
      and NCES tuition averages into a unified panel indexed by [Country_Code, Academic_Level, Year].
    """
    print("--- Constructing Master Panel Dataset ---")
    
    # 1. Reshape IIE Academic Level enrollment data from wide to long format
    value_vars = [c for c in academic_level_df.columns if "Intl_Students" in c]
    melted_enroll = academic_level_df.melt(
        id_vars=[c for c in academic_level_df.columns if c not in value_vars],
        value_vars=value_vars,
        var_name="Year_Metric",
        value_name="Enrollment"
    )
    melted_enroll["Year"] = melted_enroll["Year_Metric"].str.extract(r"(\d{4}-\d{2})")
    melted_enroll["Year_Start"] = melted_enroll["Year"].str.slice(0, 4).astype(float)
    melted_enroll["Enrollment"] = pd.to_numeric(melted_enroll["Enrollment"], errors="coerce")
    
    # 2. Harmonize NCES Tuition (tabn330_df) with Academic Years
    tabn330_df["Year_Clean"] = tabn330_df["Year"].astype(str).str.strip()
    extracted_years = tabn330_df["Year_Clean"].str.slice(0, 4)
    tabn330_df["Year_Start"] = pd.to_numeric(extracted_years, errors="coerce")
    
    # Merge Enrollment with NCES Tuition on Academic Year Start
    panel_df = pd.merge(
        melted_enroll, 
        tabn330_df[["Year_Start", "Average_Total", "Average_Public"]], 
        on="Year_Start", 
        how="left"
    )
    
    # 3. Merge macroeconomic drivers from IMF_clean (GDP per capita, CPI inflation)
    # Assuming IMF_clean has columns like ['Country_Code', 'Year', 'NGDPDPC', 'PCPIE']
    if not IMF_clean.empty and "Country_Code" in IMF_clean.columns:
        panel_df = pd.merge(
            panel_df,
            IMF_clean,
            on=["Country_Code", "Year_Start"],
            how="left"
        )
    
    print(f"Master panel shape after merging: {panel_df.shape}")
    return panel_df


def apply_lags_and_transformations(panel_df):
    """
    TODO: 3. Handle Time-Series Lags (Economic Decision Delay)
    - Create 1-year and 2-year lagged features for home-country GDP per capita, inflation, and USD exchange rates.
    
    TODO: 4. Log-Log Transformations & Elasticity Features
    - Apply natural logarithms (np.log) to monetary and volume variables (Enrollment, GDP per capita, Tuition).
    - Compute year-over-year percentage growth rates for macroeconomic volatility metrics.
    """
    print("--- Applying Time-Series Lags & Log-Log Transformations ---")
    
    # 1. Update sort keys to use Entity_ID instead of Country_Code
    sort_keys = [c for c in ["Entity_ID", "Year_Start"] if c in panel_df.columns]
    panel_df = panel_df.sort_values(sort_keys).reset_index(drop=True)
    
    # 2. Group by Entity_ID so shifts happen within each academic level over time
    group_cols = [c for c in ["Entity_ID"] if c in panel_df.columns]
    
    # Ensure target lag columns are fully numeric
    lag_targets = [col for col in ["NGDPDPC", "PCPIE", "Average_Total"] if col in panel_df.columns]
    for col in lag_targets:
        panel_df[col] = pd.to_numeric(panel_df[col], errors="coerce")
        if group_cols:
            panel_df[f"{col}_lag1"] = panel_df.groupby(group_cols)[col].shift(1)
            panel_df[f"{col}_lag2"] = panel_df.groupby(group_cols)[col].shift(2)
        else:
            panel_df[f"{col}_lag1"] = panel_df[col].shift(1)
            panel_df[f"{col}_lag2"] = panel_df[col].shift(2)

    # 4. Log-Log Transformations (for direct economic elasticity interpretation)
    log_targets = ["Enrollment", "Average_Total", "NGDPDPC"]
    for col in log_targets:
        if col in panel_df.columns:
            # Force numeric conversion so stray strings or symbols become NaN instead of crashing
            numeric_series = pd.to_numeric(panel_df[col], errors="coerce")
            # Add small epsilon or handle NaNs gracefully via clip
            panel_df[f"log_{col}"] = np.log(numeric_series.clip(lower=1e-5))
            
    # Compute Year-over-Year Growth Rates grouped by Entity_ID
    if "Average_Total" in panel_df.columns:
        panel_df["Tuition_YoY_Growth"] = (
            panel_df.groupby(group_cols)["Average_Total"].pct_change(fill_method=None) 
            if group_cols else panel_df["Average_Total"].pct_change()
        )

    return panel_df


def export_master_features(panel_df):
    """
    TODO: 5. Export Master Feature Set
    - Save the finalized panel dataframe to 'data/master_panel_features.csv' for modeling.
    """
    os.makedirs("data", exist_ok=True)
    output_path = "data/master_panel_features.csv"
    panel_df.to_csv(output_path, index=False)
    print(f"Successfully exported master feature set to {output_path}")