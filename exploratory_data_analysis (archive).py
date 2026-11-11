# ==========================================
# FILE: exploratory_data_analysis.py
# PURPOSE: Exploratory Data Analysis (EDA) and 
#          trend visualization for international 
#          student mobility and economic drivers.
# ==========================================

# TODO: 1. Load Cleaned Datasets
#       - Import cleaned dataframes from get_data.py (Open Doors enrollment, NCES tuition, WDI/IMF macro indicators).

# TODO: 2. Temporal Enrollment Trend Analysis
#       - Aggregate total international student enrollment counts by academic level (Associate's, Bachelor's, Master's, Doctoral) 
#         across years (1999–2025).
#       - Plot enrollment trajectories over time using matplotlib/seaborn to highlight pre-pandemic peaks, 
#         COVID-19 contractions, and post-pandemic recoveries.

# TODO: 3. Tuition vs. Enrollment Dual-Axis Visualization
#       - Plot U.S. average undergraduate/graduate tuition costs (from NCES Digest Table 330.50) 
#         alongside total international student enrollment counts on dual axes to visually inspect co-movement or divergence.

# TODO: 4. Macroeconomic Correlation Matrix
#       - Merge origin-country GDP per capita and inflation metrics with country-level enrollment shares.
#       - Generate a correlation heatmap (seaborn.heatmap) to evaluate raw bivariate relationships 
#         between economic proxies and student mobility before econometric modeling.

# TODO: 5. Save Summary Statistics & Figures
#       - Export summary statistics tables (mean, median, std, min, max) for key variables.
#       - Save generated plots as high-resolution PNGs in an 'output/' or 'figures/' directory.

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def run_eda(
    academic_level_df,
    cost1_df,
    WDICountry_df,
    IMF_df,
    drvcost_df,
    drvef_df,
    tabn330_df,
):
  print("--- Starting Exploratory Data Analysis (EDA) ---")

  os.makedirs("figures", exist_ok=True)
  sns.set_theme(style="whitegrid")

  if not academic_level_df.empty:
    value_vars = [c for c in academic_level_df.columns if "Intl_Students" in c]
    if value_vars:
      melted_df = academic_level_df.melt(
          id_vars=["Academic_Level"],
          value_vars=value_vars,
          var_name="Year_Metric",
          value_name="Enrollment",
      )
      melted_df["Year"] = melted_df["Year_Metric"].str.extract(r"(\d{4}-\d{2})")

      plt.figure(figsize=(14, 7))
      sns.lineplot(
          data=melted_df,
          x="Year",
          y="Enrollment",
          hue="Academic_Level",
          marker="o",
          linewidth=2.5,
      )
      plt.title(
          "International Student Enrollment Trends by Academic Level"
          " (1999–2025)",
          fontsize=14,
          fontweight="bold",
      )
      plt.xlabel("Academic Year", fontsize=12)
      plt.ylabel("International Student Count", fontsize=12)
      plt.xticks(rotation=45, ha="right")
      plt.legend(
          title="Academic Level", bbox_to_anchor=(1.05, 1), loc="upper left"
      )
      plt.tight_layout()

      fig_path = "figures/enrollment_trends_by_level.png"
      plt.savefig(fig_path, dpi=300)
      print(f"Saved enrollment trend plot to {fig_path}")
      plt.close()

  if not tabn330_df.empty and "Average_Total" in tabn330_df.columns:
    plt.figure(figsize=(12, 6))
    plt.plot(
        tabn330_df["Year"],
        tabn330_df["Average_Total"],
        label="Average Total Tuition",
        marker="s",
        color="b",
    )
    if "Average_Public" in tabn330_df.columns:
      plt.plot(
          tabn330_df["Year"],
          tabn330_df["Average_Public"],
          label="Average Public Tuition",
          marker="^",
          color="g",
      )

    plt.title(
        "U.S. Institutional Tuition Costs Over Time (NCES Digest)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Academic Year", fontsize=12)
    plt.ylabel("Average Tuition (USD)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()

    fig_path = "figures/nces_tuition_trends.png"
    plt.savefig(fig_path, dpi=300)
    print(f"Saved tuition trend plot to {fig_path}")
    plt.close()

  print("--- EDA Pipeline Complete ---")