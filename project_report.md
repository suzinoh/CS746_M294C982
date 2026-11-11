# Quantifying the Impact of Economic Shocks and Tuition Costs on Cross-Border Student Flows
__Suzin Oh / M294C982__

## 1. Research Objectives
- **Primary Research Problem & Significance:** 
    International student enrollment in U.S. colleges changes a lot due to global economic shifts, currency values, and rising tuition costs. Understanding these economic factors matters because students and families look closely at home-country wealth and college expenses before deciding to study abroad. Knowing how these factors work helps schools and leaders prepare for sudden changes in enrollment.
- **Overall Goal:** 
    To build a reliable economic model that uncovers and measures the main financial factors driving international student enrollment trends over time.
- **Specific Research Objectives:**
    1. Data Collection and Cleanup: Combine U.S. college enrollment numbers with global economic data (like income, inflation, and tuition costs) into one clean system.

    2. Smart Data Adjustments: Use math transformations and add 1-year or 2-year time delays to account for the fact that students usually make their college plans well before they actually enroll.

    3. Model Testing: Build and test regression models that capture the unique differences between various student levels (such as undergraduate versus graduate students) to ensure accurate predictions.
- **Expected Scientific/Technical Contributions:** 
    1. Clear insights into how much international student numbers change when home-country wealth or U.S. tuition prices go up or down.

    2. Proving that breaking data down by academic level works much better for analysis than looking at global totals all at once.

    3. Creating a reusable, automated Python data pipeline for future education research.
---

## 2. Materials and Methods

### 2.1 Dataset
- **Dataset Name & Source:** IIE Open Doors, US Dept of State Visa Stats, World Bank WDI, IMF, NCES IPEDS.
- **Dataset Description:** A panel dataset tracking international student enrollment volumes across distinct academic levels (Freshman, Sophomore, Junior, Senior, Graduate, etc.) matched with annual macroeconomic indicators and tuition benchmarks over multiple years.
  
- **Data Collection:** Publicly available panel datasets compiled and merged via an automated Python ingestion pipeline (see file `get_data.py`).

- **Data Partitioning:** An 80/20 train-test split (chronological/randomized split across panel entities) to evaluate both in-sample fit and out-of-sample generalization.

### 2.2 Data Preprocessing
Processed in file `get_data.py`.

- **Data cleaning and missing-value handling:** Automated cleaning scripts (get_data.py) addressing missing strings (---N), enforcing numeric typing via pd.to_numeric(..., errors="coerce"), and handling missing macroeconomic indicators.
- **Noise removal and signal/image enhancement:** Standardization of monetary values and alignment of multi-source time-series datasets into a unified annual index.
- **Normalization or standardization:** Application of natural logarithm transformations ($\ln$) to monetary and volume variables to stabilize variance and linearize exponential growth trends.
- **Segmentation, resizing, or resampling:** Harmonization of temporal indexes and grouping of panel units by academic level (Entity_ID) to establish structured time-series panels.

### 2.3 Feature Extraction and Selection
- **Input Features:** Enrollment counts, average total tuition costs (Average_Total), home-country GDP per capita (NGDPDPC), consumer price index inflation (PCPIE), and currency metrics.
- **Feature Extraction:** Construction of 1-year and 2-year time-series lags (_lag1, _lag2) for macroeconomic drivers to account for the decision-making and application delay window of international students.
- **Feature Selection:** Selection of core economic indicators representing tuition burden, home-country purchasing power, and macroeconomic stability.
- **Feature Representation:** Finalized panel feature set exported to data/master_panel_features.csv.

### 2.4 Proposed Models and Methodology
- **Baseline Models:** Pooled OLS Regression:
Estimates a single intercept across all data without accounting for differences between academic tiers. It serves as a benchmark to measure how much explanatory power is gained by adding panel structure.
- **Proposed Model:** Academic-Level Fixed Effects (FE) Regression:Adds categorical dummy variables ($C(\text{Entity\_ID})$) for each academic level (Freshman, Graduate, etc.). This model was chosen because enrollment behavior varies significantly by academic stage; adding fixed effects absorbs these tier-specific baseline differences and isolates the true effect of macroeconomic variables (log_Average_Total, log_NGDPDPC, PCPIE).
- **Training Strategy:** The model is trained on 80% of the dataset and evaluated on the remaining 20% test split (random_state=42). While standard econometrics uses 100% of data for inference, an out-of-sample split was selected to test predictive accuracy and ensure the model generalizes without overfitting.
- **Evaluation Metrics:**
  - Training $R^2$: Measures the proportion of enrollment variance explained by the model in-sample.
  - Out-of-Sample RMSE: Evaluates prediction error on unseen test data on a log scale
  - Coefficient p-values: Assesses the statistical significance of each economic feature.

---

## 3. Primary Results
- **Main Experimental Findings:** The incorporation of academic-level fixed effects drastically improved model explanatory power, shifting $R^2$ from near-zero global aggregates up to 0.8595.
- **Comparison with Baseline Methods:** While Pooled OLS failed to capture structural differences between academic tiers, the Academic-Level Fixed Effects model successfully accounted for tier-specific enrollment baselines.
- **Quantitative Improvements & Statistical Significance:**
  - Training $R^2$: 0.8595 (explaining ~86% of enrollment variance).
  - Out-of-Sample Test RMSE: 0.2674 (translating to high predictive accuracy on log-transformed student volumes).
- **Figures & Visualizations:** 
  - `figures/enrollment_trends_by_level.png`
  - `figures/nces_tuition_trends.png`

---

## 4. Discussion
- **Interpretation of Findings:** Log-log specifications allow direct interpretation of economic elasticities, showing how sensitive enrollment decisions across different academic tiers are to tuition inflation and home-country GDP shifts.
- **Methodology Performance:** Academic-level fixed effects successfully isolate unobserved structural traits unique to each educational stage (e.g., undergraduate cost sensitivity versus graduate research funding reliance).
- **Scientific Contributions & Practical Applications:** Demonstrates that structural panel design outperforms global aggregate modeling in higher education forecasting, offering university administrators a reliable tool for enrollment planning and tuition pricing strategy.
- **Limitations & Challenges:** Unmeasured external shocks such as visa processing backlogs, sudden geopolitical shifts, and post-graduation policy changes (e.g., OPT rules) remain uncaptured by macroeconomic indicators alone.

---

## 5. Conclusion
- **Summary of Contributions:** Successfully developed an end-to-end econometric pipeline modeling international student mobility through academic-level panel data and time-series lags.
- **Key Findings:** Proved that economic drivers combined with academic-level fixed effects achieve an 85.95% fit on training data and robust out-of-sample predictive performance ($\text{RMSE} = 0.2674$).
- **Broader Significance:** Highlights the necessity of granular panel data structuring when evaluating macro-level educational trends.
- **Future Directions:** Integrating monthly visa refusal logs, exchange rate fluctuations, and institutional financial aid data to further refine predictive granularity.
---

## 6. Current Project Status and Next Steps
- **Current Status:** Model Evaluation & Final Report Preparation.
- **Completed Milestones:**
  - Data ingestion pipeline (`get_data.py`)
  - Exploratory data analysis (`eda.py`)
  - Panel feature engineering & lagging (`feature_eng.py`)
  - Econometric modeling (`modeling.py`)
- **Ongoing Activities:** Report drafting and finalizing regression tables.
- **Expected Deliverables:** Reproducible Python codebase, processed panel dataset, and final research report.
- **Next Milestones:** Submission of final project deliverable.