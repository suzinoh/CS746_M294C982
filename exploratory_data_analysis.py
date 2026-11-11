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
  os.makedirs("output", exist_ok=True)
  sns.set_theme(style="whitegrid")

  # ---------------------------------------------------------
  # TODO: 1. Load Cleaned Datasets (Handled via parameters)
  # ---------------------------------------------------------
  print("[1/5] Datasets successfully loaded into EDA pipeline.")

  # ---------------------------------------------------------
  # TODO: 2. Temporal Enrollment Trend Analysis
  # ---------------------------------------------------------
  melted_enroll_df = pd.DataFrame()
  if not academic_level_df.empty:
    value_vars = [c for c in academic_level_df.columns if "Intl_Students" in c]
    if value_vars:
      melted_enroll_df = academic_level_df.melt(
          id_vars=["Academic_Level"],
          value_vars=value_vars,
          var_name="Year_Metric",
          value_name="Enrollment",
      )
      melted_enroll_df["Year"] = melted_enroll_df["Year_Metric"].str.extract(r"(\d{4}-\d{2})")
      melted_enroll_df["Enrollment"] = pd.to_numeric(melted_enroll_df["Enrollment"], errors="coerce")
      melted_enroll_df = melted_enroll_df.dropna(subset=["Year", "Enrollment"])

      major_levels = [
          "Undergraduate", 
          "Graduate", 
          "Associate's", 
          "Bachelor's", 
          "Master's", 
          "Doctoral"
      ]
      filtered_enroll_df = melted_enroll_df[
          melted_enroll_df["Academic_Level"].str.contains("|".join(major_levels), case=False, na=False)
      ]
      if filtered_enroll_df.empty:
        filtered_enroll_df = melted_enroll_df

      fig, ax = plt.subplots(figsize=(12, 6))
      sns.lineplot(
          data=filtered_enroll_df,
          x="Year",
          y="Enrollment",
          hue="Academic_Level",
          marker="o",
          linewidth=2.5,
          ax=ax
      )
      ax.set_title(
          "International Student Enrollment Trends by Major Academic Level (1999–2025)",
          fontsize=13,
          fontweight="bold",
          pad=15
      )
      ax.set_xlabel("Academic Year", fontsize=11, labelpad=8)
      ax.set_ylabel("International Student Count", fontsize=11, labelpad=8)
      plt.xticks(rotation=45, ha="right")
      ax.legend(
          title="Academic Level",
          bbox_to_anchor=(1.02, 1),
          loc="upper left",
          frameon=True,
          fontsize=10
      )
      ax.yaxis.set_major_formatter('{x:,.0f}')
      fig.subplots_adjust(right=0.78, bottom=0.2)
      
      fig_path = "figures/enrollment_trends_by_level.png"
      plt.savefig(fig_path, dpi=300)
      plt.close(fig)
      print(f"[2/5] Saved temporal enrollment trend plot to {fig_path}")

  # ---------------------------------------------------------
  # TODO: 3. Tuition vs. Enrollment Dual-Axis Visualization
  # ---------------------------------------------------------
  if not tabn330_df.empty and not melted_enroll_df.empty and "Average_Total" in tabn330_df.columns:
    clean_t_df = tabn330_df.copy()
    clean_t_df["Year_Clean"] = clean_t_df["Year"].astype(str).str.strip()
    clean_t_df = clean_t_df[
        clean_t_df["Year_Clean"].str.contains(r"^\d{4}-\d{2,4}$", regex=True, na=False)
    ].copy()
    clean_t_df["Average_Total"] = pd.to_numeric(clean_t_df["Average_Total"], errors="coerce")
    clean_t_df = clean_t_df.dropna(subset=["Average_Total"])
    clean_t_df["Start_Year"] = clean_t_df["Year_Clean"].str.slice(0, 4).astype(int)
    clean_t_df = clean_t_df.drop_duplicates(subset=["Start_Year"]).sort_values("Start_Year").reset_index(drop=True)

    # Aggregate total international enrollment per academic year across all levels
    total_enroll_annual = melted_enroll_df.groupby("Year")["Enrollment"].sum().reset_index()
    total_enroll_annual["Start_Year"] = total_enroll_annual["Year"].str.slice(0, 4).astype(int)

    # Merge tuition and aggregate enrollment on Start_Year
    merged_dual = pd.merge(clean_t_df, total_enroll_annual, on="Start_Year", how="inner").sort_values("Start_Year")

    if not merged_dual.empty:
      fig, ax1 = plt.subplots(figsize=(12, 6))

      color = "#1f77b4"
      ax1.set_xlabel("Academic Year", fontsize=11, labelpad=8)
      ax1.set_ylabel("Average Total U.S. Tuition (USD)", color=color, fontsize=11, fontweight="bold")
      line1 = ax1.plot(merged_dual["Year_Clean"], merged_dual["Average_Total"], color=color, marker="s", linewidth=2.5, label="Average Total Tuition")
      ax1.tick_params(axis="y", labelcolor=color)
      ax1.yaxis.set_major_formatter('${x:,.0f}')
      plt.xticks(rotation=45, ha="right")

      ax2 = ax1.twinx()  
      color = "#ff7f0e"
      ax2.set_ylabel("Total International Enrollment", color=color, fontsize=11, fontweight="bold")
      line2 = ax2.plot(merged_dual["Year_Clean"], merged_dual["Enrollment"], color=color, marker="o", linewidth=2.5, label="Total International Enrollment")
      ax2.tick_params(axis="y", labelcolor=color)
      ax2.yaxis.set_major_formatter('{x:,.0f}')

      lines = line1 + line2
      labels = [l.get_label() for l in lines]
      ax1.legend(lines, labels, loc="upper left", frameon=True)

      plt.title("U.S. Tuition Costs vs. Total International Student Enrollment (Dual-Axis)", fontsize=13, fontweight="bold", pad=15)
      fig.tight_layout()
      
      dual_fig_path = "figures/tuition_vs_enrollment_dual_axis.png"
      plt.savefig(dual_fig_path, dpi=300)
      plt.close(fig)
      print(f"[3/5] Saved Tuition vs Enrollment dual-axis plot to {dual_fig_path}")

  # ---------------------------------------------------------
  # TODO: 4. Macroeconomic Correlation Matrix
  # ---------------------------------------------------------
  if not IMF_df.empty:
    numeric_imf = IMF_df.select_dtypes(include=[np.number])
    if not numeric_imf.empty:
      corr_matrix = numeric_imf.corr()
      
      fig, ax = plt.subplots(figsize=(10, 8))
      sns.heatmap(corr_matrix.iloc[:10, :10], annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
      ax.set_title("Macroeconomic Indicators Correlation Matrix (IMF Proxies)", fontsize=13, fontweight="bold", pad=15)
      plt.tight_layout()
      
      corr_fig_path = "figures/macro_correlation_heatmap.png"
      plt.savefig(corr_fig_path, dpi=300)
      plt.close(fig)
      print(f"[4/5] Saved macroeconomic correlation heatmap to {corr_fig_path}")

  # ---------------------------------------------------------
  # TODO: 5. Save Summary Statistics & Figures
  # ---------------------------------------------------------
  summary_stats_list = []
  
  if not tabn330_df.empty:
    t_desc = tabn330_df[["Average_Total", "Average_Public"]].describe().T
    summary_stats_list.append(t_desc)

  if not melted_enroll_df.empty:
    e_desc = melted_enroll_df[["Enrollment"]].describe().T
    summary_stats_list.append(e_desc)

  if summary_stats_list:
    master_summary = pd.concat(summary_stats_list)
    summary_csv_path = "output/summary_statistics_key_variables.csv"
    master_summary.to_csv(summary_csv_path)
    print(f"[5/5] Exported summary statistics table to {summary_csv_path}")

  print("--- EDA Pipeline Complete ---")