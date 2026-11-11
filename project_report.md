# Quantifying the Impact of Economic Shocks and Tuition Costs on Cross-Border Student Flows
__Suzin Oh / M294C982__

## 1. Research Objectives
- **Primary Research Problem & Significance:** 
  [Describe the shift in international student enrollment and why economic drivers matter.]
- **Overall Goal:** 
  [State the overarching goal of identifying external structural drivers governing mobility.]
- **Specific Research Objectives:**
  1. [Objective 1]
  2. [Objective 2]
  3. [Objective 3]
- **Expected Scientific/Technical Contributions:** 
  [Explain contributions to economic modeling of higher education mobility.]

---

## 2. Materials and Methods

### 2.1 Dataset
- **Dataset Name & Source:** IIE Open Doors, US Dept of State Visa Stats, World Bank WDI, IMF, NCES IPEDS.
- **Dataset Description:** [Type, size, number of samples, classes, and modalities.]
- **Data Collection:** [Publicly available panel datasets.]
- **Data Partitioning:** [Training, validation, and testing strategy / chronological split.]

### 2.2 Data Preprocessing
- **Data cleaning and missing-value handling:** [Describe cleaning steps in get_data.py.]
- **Noise removal and signal/image enhancement:** [N/A or currency/inflation standardization.]
- **Normalization or standardization:** [Log-log transformations and scaling.]
- **Data augmentation and class imbalance handling:** [As applicable.]
- **Segmentation, resizing, or resampling:** [Time-series alignment.]

### 2.3 Feature Extraction and Selection
- **Input Features:** [Enrollment counts, tuition costs, GDP per capita, CPI inflation, exchange rates.]
- **Feature Extraction:** [Time-series lags (1-year/2-year) for decision delay windows.]
- **Feature Selection:** [Dimensionality reduction or core economic indicator selection.]
- **Feature Representation:** [Final panel feature set saved in data/master_panel_features.csv.]

### 2.4 Proposed Models and Methodology
- **Baseline Models:** [Pooled Ordinary Least Squares (OLS) regression.]
- **Proposed Model:** [Country Fixed Effects (FE) panel regression to control for unobserved heterogeneity.]
- **Training Strategy:** [Hyperparameters, optimization, and time-series cross-validation.]
- **Evaluation Metrics:** [R-squared, Adjusted R-squared, RMSE, p-values.]

---

## 3. Primary Results
- **Main Experimental Findings:** [Summarize econometric regression outputs.]
- **Comparison with Baseline Methods:** [Compare Pooled OLS vs. Country Fixed Effects performance.]
- **Quantitative Improvements & Statistical Significance:** [Report coefficients, standard errors, and p-values.]
- **Figures & Visualizations:** 
  - `figures/enrollment_trends_by_level.png`
  - `figures/nces_tuition_trends.png`
- **Result Status:** [Preliminary / Finalized.]

---

## 4. Discussion
- **Interpretation of Findings:** [Relate economic elasticities (tuition, GDP, inflation) to student mobility decisions.]
- **Methodology Performance:** [Why Fixed Effects better capture unobserved country-level traits.]
- **Scientific Contributions & Practical Applications:** [Implications for university recruitment and policymaking.]
- **Limitations & Challenges:** [Unobserved visa backlogs, geopolitical shocks, policy changes.]

---

## 5. Conclusion
- **Summary of Contributions:** [Recap how macroeconomic drivers were modeled.]
- **Key Findings:** [Highlight primary elasticity insights.]
- **Broader Significance:** [Significance for global higher education trends.]
- **Future Directions:** [Next steps, integrating monthly visa refusal logs or institutional financial aid data.]

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