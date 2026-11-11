# Research Problem / Problem Statement
# Recently, shift in number of international students in the country has been observed to be changing dramatically. While there are extensive studies aggregating enrollment counts, very few studies have discussed how the economic factors that impact the decrease/increase of enrollment numbers. This project will identify and quantify the external structural drivers that governs international student mobility across the degree levels and primary origin countries. This study will pay attention to economic factors only.
#
# Dataset
#
# For this project, few public datasets are available;
#
# 1. IIE Open Doors Tabular Data; country-level annal enrollment breakdowns by academic level
#    link: https://opendoorsdata.org/?download=https://opendoorsdata.org/wp-content/uploads/2025/11/Census_Academic-Level_OD25_Website.xlsx
#          for historican data INTERNATIONAL STUDENTS BY DETAILED ACADEMIC LEVEL, 1999/00 - 2024/25
# 2. US Department of state visa statistics; shows monthly F-1 visa issuance and refual logs per embassy
#    link: way to calculate the refusal https://travel.state.gov/content/travel/en/legal/visa-law0/visa-statistics/nonimmigrant-visa-statistics/nonimmigrant-b-visa-adjusted-refusal-rates-by-nationality.html
#    link: for pdfs (need to process it) https://travel.state.gov/content/travel/en/legal/visa-law0/visa-statistics/nonimmigrant-visa-statistics/monthly-nonimmigrant-visa-issuances.html
# 3. World Bank & IMF Macroeconomic Indicators : country-level economic indicators such as foreign GDP per capita, USD exchange rate, local CPI inflation
#    link: (downloaded in data/WDI_CSV) https://datatopics.worldbank.org/world-development-indicators/
#    core repository for country-level GDP per capita (constant USD), local consumer price index (CPI) inflation, and exchange rates.
#    link: IMF.csv https://data.imf.org/en/Data-Explorer?datasetUrn=IMF.RES:WEO(9.0.0)
# 4. NCES Database; average tuition costs, institutional characteristics across U.S. host regions
#    link: https://nces.ed.gov/ipeds/Search?query=tuition&query2=tuition&resultType=table&page=2&sortBy=relevance&overlayDigestTableId=202676
#    link: https://nces.ed.gov/ipeds/datacenter/DataFiles.aspx?year=-1&surveyNumber=19&sid=6374a433-25ff-46bd-939c-5f5c4c3efe96&rtid=7 
#
# Above datasets are all downloaded in the folder data/ and shall be processed in get_data.py for further processing

import pandas as pd
import numpy as np
import os as os

def load_data():
    """
    Load and preprocess the datasets required for the project.
    Returns:
        academic_level_df (pd.DataFrame): Preprocessed academic level data.
        cost1_df (pd.DataFrame): Preprocessed cost data.
        WDICountry_df (pd.DataFrame): Preprocessed WDI country data.
        IMF_df (pd.DataFrame): Preprocessed IMF data.
        drvcost_df (pd.DataFrame): Preprocessed driver cost data.
        drvef_df (pd.DataFrame): Preprocessed driver effect data.
        tabn330_df (pd.DataFrame): Preprocessed tabn330 data.
    """
    # step 1: reading Census_Academic-Level_OD25_Website.xlsx
    # step 1.a: Read raw excel file with NO headers to safely control row slicing
    file_path = "data/Census_Academic-Level_OD25_Website.xlsx"
    df_raw = pd.read_excel(file_path, header=None, engine="openpyxl")

    # step 1.b: Extract rows 1 and 2 (0-indexed) for Years and Metrics based on your Excel layout
    year_row = df_raw.iloc[1, :]
    metric_row = df_raw.iloc[2, :]

    # Forward-fill the merged year headers across columns
    filled_years = pd.Series(year_row).ffill()

    # step 1.c: Construct unique flat column names
    new_cols = []
    for idx, (y, m) in enumerate(zip(filled_years, metric_row)):
        m_str = str(m).strip()

        # Identify the Academic Level label column
        if "Academic Level" in m_str or idx == 1:
            new_cols.append("Academic_Level")
            continue

        y_str = str(y).strip().replace("/", "-")

        if pd.isna(y) or "Unnamed" in y_str or y_str == "nan":
            new_cols.append(f"Drop_Col_{idx}")
        else:
            if "Int'l" in m_str or "Students" in m_str:
                metric_clean = "Intl_Students"
            elif "%" in m_str:
                metric_clean = "Pct_Total"
            else:
                metric_clean = f"Metric_{idx}"
            new_cols.append(f"{y_str}_{metric_clean}")

    df_raw.columns = new_cols

    # step 1.d: Drop duplicate columns and spacer columns to prevent DataFrame expansion bugs
    df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()]
    df_raw = df_raw.loc[:, ~df_raw.columns.str.startswith("Drop_Col_")]
    df_raw = df_raw.loc[:, ~df_raw.columns.str.contains("Metric")]

    # step 1.e: Slice actual data starting from row 4 onwards where data begins
    academic_level_df = df_raw.iloc[4:].copy()

    # step 1.f: Drop rows where Academic_Level is missing/NaN
    academic_level_df = academic_level_df.dropna(subset=["Academic_Level"])

    # step 1.g: Clean rows safely: drop footnotes/metadata rows
    academic_level_df = academic_level_df[
        ~academic_level_df["Academic_Level"]
        .astype(str)
        .str.contains("NOTE|SOURCE|Total", case=False, na=False)
    ]

    # step 1.h: Convert metric columns to numeric (stripping commas, dashes, and whitespace)
    for col in academic_level_df.columns[1:]:
        academic_level_df[col] = (
            academic_level_df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("-", "", regex=False)
            .str.strip()
        )
        academic_level_df[col] = pd.to_numeric(
            academic_level_df[col], errors="coerce"
    )

    # Reset index and display preview
    academic_level_df = academic_level_df.reset_index(drop=True)
    print(academic_level_df.head())

    # step 2: reading cost1_2024.csv
    cost1_df = pd.read_csv('data/cost1_2024.csv')
    print(cost1_df.head(15))

    # step 3: reading WDI_CSV/WDICountry.csv
    WDICountry_df = pd.read_csv('data/WDI_CSV/WDICountry.csv')
    print(WDICountry_df.head(15))

    # step 4: reading IMF.csv
    IMF_df = pd.read_csv('data/IMF.csv')
    print(IMF_df.head(15))

    # step 5: reading drvcost2024.csv
    drvcost_df = pd.read_csv('data/drvcost2024.csv')
    print(drvcost_df.head(15))

    # step 6: reading drvef2024.csv
    drvef_df = pd.read_csv('data/drvef2024.csv')
    print(drvef_df.head(15))

    # step 7: reading tabn330.50.xlsx
    # step 7.a: Load the dataframe skipping the title and header rows (rows 0 to 6)
    # Based on the image, the data rows begin at row index 7 in Excel (skiprows=6)
    tabn330_df = pd.read_excel(
        "data/tabn330.50.xlsx", skiprows=6, header=None
    )

    # step 7.b: Keep only the 12 columns corresponding to columns A through L
    tabn330_df = tabn330_df.iloc[:, :12]

    # step 7.c: Assign explicit column names matching the 12 columns in your image
    tabn330_df.columns = [
        "Year",
        "Average_Total",
        "Average_Public",
        "Average_Private_Total",
        "Average_Private_Nonprofit",
        "Average_Private_For_profit",
        "Public_25th_Percentile",
        "Public_Median",
        "Public_75th_Percentile",
        "Private_Nonprofit_25th_Percentile",
        "Private_Nonprofit_Median",
        "Private_Nonprofit_75th_Percentile",
    ]

    # step 7.d: Clean missing values ('---') and Excel overflow artifacts ('######')
    tabn330_df = tabn330_df.replace(
        [r"^\s*---\s*$", r"^\s*#+\s*$"], np.nan, regex=True
    )

    # step 7.e: Strip currency symbols, commas, and convert metric columns to numeric floats
    value_cols = tabn330_df.columns[1:]
    for col in value_cols:
        tabn330_df[col] = (
            tabn330_df[col]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
    )
    tabn330_df[col] = pd.to_numeric(tabn330_df[col], errors="coerce")

    # step 7.f: Drop trailing footer/note rows if any exist below the table
    tabn330_df = tabn330_df.dropna(subset=["Year"])

    print(tabn330_df.head(15))
    
    return academic_level_df, cost1_df, WDICountry_df, IMF_df, drvcost_df, drvef_df, tabn330_df


def clean_and_prepare_panel_data(
    academic_level_df,
    cost1_df,
    WDICountry_df,
    IMF_df,
    drvcost_df,
    drvef_df,
    tabn330_df,
):
  print("--- Starting Post-Load Cleaning & Harmonization ---")

  # 1. Clean WDI Country metadata (ISO-3 code standardization)
  if not WDICountry_df.empty and "Country Code" in WDICountry_df.columns:
    WDICountry_clean = WDICountry_df.dropna(subset=["Country Code"]).copy()
    # Filter out non-country aggregate rows if needed (e.g., regions like 'WLD', 'LDC')
    WDICountry_clean = WDICountry_clean[
        ~WDICountry_clean["Region"].isna()
    ].copy()
  else:
    WDICountry_clean = pd.DataFrame()

  # 2. Clean IMF Macroeconomic panel
  if not IMF_df.empty:
    # Replace suppression strings and cast year columns to float
    IMF_df.replace(["..", "-", "N/A", ""], np.nan, inplace=True)
    year_cols = [
        c
        for c in IMF_df.columns
        if str(c).isdigit() or (len(str(c)) == 4 and str(c).startswith("20"))
    ]
    for col in year_cols:
      IMF_df[col] = pd.to_numeric(
          IMF_df[col].astype(str).str.replace(",", "").str.strip(),
          errors="coerce",
      )

    # Filter for core economic indicators relevant to student mobility (GDP per capita, Inflation)
    core_series = ["PPPPC", "PCPIE", "NGDPDPC"]
    if "SERIES_CODE" in IMF_df.columns:
      IMF_clean = IMF_df[
          IMF_df["SERIES_CODE"]
          .str.contains("|".join(core_series), case=False, na=False)
      ].copy()
    else:
      IMF_clean = IMF_df.copy()
  else:
    IMF_clean = pd.DataFrame()

  # 3. Clean IPEDS institutional cost files
  ipeds_list = []
  for df_obj in [cost1_df, drvcost_df, drvef_df]:
    if not df_obj.empty:
      df_obj.replace(
          [r"^\s*\.\.\s*$", r"^\s*---\s*$", r"^\s*-\s*$"],
          np.nan,
          regex=True,
          inplace=True,
      )
      ipeds_list.append(df_obj)

  print("--- Post-Load Cleaning Complete ---")
  return WDICountry_clean, IMF_clean, ipeds_clean_merged if 'ipeds_clean_merged' in locals() else None