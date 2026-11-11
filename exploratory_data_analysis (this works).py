# ==========================================
# FILE: eda.py
# PURPOSE: Exploratory Data Analysis (EDA) and 
#          trend visualization for international 
#          student mobility and economic drivers.
# ==========================================

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

  # ---------------------------------------------------------
  # 1. Enrollment Trends Plot
  # ---------------------------------------------------------
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
      melted_df["Enrollment"] = pd.to_numeric(
          melted_df["Enrollment"], errors="coerce"
      )
      melted_df = melted_df.dropna(subset=["Year", "Enrollment"])

      major_levels = [
          "Undergraduate",
          "Graduate",
          "Associate's",
          "Bachelor's",
          "Master's",
          "Doctoral",
      ]
      filtered_enroll_df = melted_df[
          melted_df["Academic_Level"]
          .str.contains("|".join(major_levels), case=False, na=False)
      ]
      if filtered_enroll_df.empty:
        filtered_enroll_df = melted_df

      fig, ax = plt.subplots(figsize=(12, 6))
      sns.lineplot(
          data=filtered_enroll_df,
          x="Year",
          y="Enrollment",
          hue="Academic_Level",
          marker="o",
          linewidth=2.5,
          ax=ax,
      )
      ax.set_title(
          "International Student Enrollment Trends by Major Academic Level"
          " (1999–2025)",
          fontsize=13,
          fontweight="bold",
          pad=15,
      )
      ax.set_xlabel("Academic Year", fontsize=11, labelpad=8)
      ax.set_ylabel("International Student Count", fontsize=11, labelpad=8)
      plt.xticks(rotation=45, ha="right")
      ax.legend(
          title="Academic Level",
          bbox_to_anchor=(1.02, 1),
          loc="upper left",
          frameon=True,
          fontsize=10,
      )

      ax.yaxis.set_major_formatter("{x:,.0f}")
      fig.subplots_adjust(right=0.78, bottom=0.2)

      fig_path = "figures/enrollment_trends_by_level.png"
      plt.savefig(fig_path, dpi=300)
      print(f"Saved clean enrollment trend plot to {fig_path}")
      plt.close(fig)

  # ---------------------------------------------------------
  # 2. NCES Tuition Trends Plot (Chronological Sorting & Deduplication)
  # ---------------------------------------------------------
  if not tabn330_df.empty and "Average_Total" in tabn330_df.columns:
    clean_t_df = tabn330_df.copy()

    clean_t_df["Year_Clean"] = clean_t_df["Year"].astype(str).str.strip()
    clean_t_df = clean_t_df[
        clean_t_df["Year_Clean"].str.contains(
            r"^\d{4}-\d{2,4}$", regex=True, na=False
        )
    ].copy()

    for col in ["Average_Total", "Average_Public"]:
      if col in clean_t_df.columns:
        clean_t_df[col] = pd.to_numeric(clean_t_df[col], errors="coerce")

    clean_t_df = clean_t_df.dropna(subset=["Average_Total"])

    # Extract start year integer, drop duplicates, and sort chronologically
    clean_t_df["Start_Year"] = (
        clean_t_df["Year_Clean"].str.slice(0, 4).astype(int)
    )
    clean_t_df = (
        clean_t_df.drop_duplicates(subset=["Start_Year"])
        .sort_values("Start_Year")
        .reset_index(drop=True)
    )

    if not clean_t_df.empty:
      fig, ax = plt.subplots(figsize=(12, 6))

      ax.plot(
          clean_t_df["Year_Clean"],
          clean_t_df["Average_Total"],
          label="Average Total Tuition",
          marker="s",
          color="#1f77b4",
          linewidth=2,
      )
      if "Average_Public" in clean_t_df.columns:
        ax.plot(
            clean_t_df["Year_Clean"],
            clean_t_df["Average_Public"],
            label="Average Public Tuition",
            marker="^",
            color="#2ca02c",
            linewidth=2,
      )

      ax.set_title(
          "U.S. Institutional Tuition Costs Over Time (NCES Digest Table"
          " 330.50)",
          fontsize=13,
          fontweight="bold",
          pad=15,
      )
      ax.set_xlabel("Academic Year", fontsize=11, labelpad=8)
      ax.set_ylabel("Average Tuition (USD)", fontsize=11, labelpad=8)

      # Space x-axis ticks cleanly
      n_ticks = len(clean_t_df)
      step = max(1, n_ticks // 12)
      plt.xticks(
          ticks=np.arange(0, n_ticks, step),
          labels=clean_t_df["Year_Clean"].iloc[::step],
          rotation=45,
          ha="right",
      )

      ax.yaxis.set_major_formatter("${x:,.0f}")
      ax.legend(loc="upper left", frameon=True, fontsize=10)
      fig.subplots_adjust(bottom=0.2, right=0.92)

      fig_path = "figures/nces_tuition_trends.png"
      plt.savefig(fig_path, dpi=300)
      print(f"Saved clean tuition trend plot to {fig_path}")
      plt.close(fig)

  print("--- EDA Pipeline Complete ---")