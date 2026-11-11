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
    TODO: 1. Harmonize Country Identifiers
    - Standardize ISO-3 country codes across WDI, IMF, and visa logs.
    - Map IIE Open Doors country names to standard ISO-3 codes.
    """
    print("--- Harmonizing Country Identifiers ---")
    # If academic_level_df contains country breakdowns, map names to ISO3 codes here.
    # Creating a dummy ISO mapping dictionary for demonstration:
    country_iso_mapping = {
        "China": "CHN",
        "India": "IND",
        "South Korea": "KOR",
        "Canada": "CAN",
        "Vietnam": "VNM",
        "Brazil": "BRA",
        "Mexico": "MEX"
    }
    
    if "Country" in academic_level_df.columns:
        academic_level_df["Country_Code"] = academic_level_df["Country"].map(country_iso_mapping).fillna("OTHER")
    else:
        # Fallback if dataset is aggregated globally or by level
        academic_level_df["Country_Code"] = "GLOBAL"
        
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
    tabn330_df["Year_Start"] = tabn330_df["Year_Clean"].str.slice(0, 4).astype(float)
    
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
    
    # Sort panel to ensure chronological ordering per group
    sort_keys = [c for c in ["Country_Code", "Academic_Level", "Year_Start"] if c in panel_df.columns]
    panel_df = panel_df.sort_values(sort_keys).reset_index(drop=True)
    
    group_cols = [c for c in ["Country_Code", "Academic_Level"] if c in panel_df.columns]
    
    # 3. Handle Time-Series Lags (1-year and 2-year lags for decision window delay)
    lag_targets = [col for col in ["NGDPDPC", "PCPIE", "Average_Total"] if col in panel_df.columns]
    for col in lag_targets:
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
            # Add small epsilon or handle NaNs gracefully
            panel_df[f"log_{col}"] = np.log(panel_df[col].clip(lower=1e-5))
            
    # Compute Year-over-Year Growth Rates
    if "Average_Total" in panel_df.columns:
        panel_df["Tuition_YoY_Growth"] = panel_df.groupby(group_cols)["Average_Total"].pct_change(fill_method=None) if group_cols else panel_df["Average_Total"].pct_change()

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


# if __name__ == "__main__":
#     from get_data import clean_and_prepare_panel_data, load_data
    
#     # Load raw datasets
#     datasets = load_data()
#     wdi_c, imf_c, ipeds_c = clean_and_prepare_panel_data(*datasets)
    
#     # Execute pipeline
#     academic_df = datasets[0]
#     harmonized_academic = harmonize_country_identifiers(academic_df, wdi_c)
    
#     master_panel = build_panel_dataset(
#         harmonized_academic, datasets[1], wdi_c, imf_c, datasets[4], datasets[5], datasets[6]
#     )
    
#     transformed_panel = apply_lags_and_transformations(master_panel)
#     export_master_features(transformed_panel)