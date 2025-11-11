# Reduced-Demand-of-Emergency-Unit
# Overview

This repository contains the code, data, and analysis from my MSc Dissertation project:
“Predictive Analysis of Emergency Unit Demand: A Data-Driven Approach Using Machine Learning” completed at Cardiff University in collaboration with Cardiff and Vale University Health Board.

The project aims to analyze three years of daily data from the Emergency Unit at the University Hospital of Wales (UHW) to identify patterns, trends, and key factors influencing demand. Insights from this analysis can help in resource planning, staff allocation, and preventive strategies to improve healthcare service efficiency.

# Objectives

To explore and visualize patterns in emergency admissions across different timeframes.

To analyze the impact of external factors such as bank holidays, weather, and temperature on healthcare demand.

To develop predictive models to forecast emergency visits and routine admissions.

To provide data-driven recommendations for operational planning and policy decisions.

# Methodology

Data Sources: Historical hospital data (Emergency Unit records) and external datasets (weather, holiday calendar).

Data Cleaning & Processing: Handled missing values, duplicates, and time-series formatting.

Exploratory Data Analysis (EDA): Identified key trends and anomalies using visual analytics.

Machine Learning Models:

Linear Regression

Random Forest Regressor

XGBoost Regressor

Prophet Time-Series Forecasting

Evaluation Metrics: MAE, RMSE, R²

Visualization: Seaborn, Matplotlib, Power BI dashboards

# Key Findings

Bank Holiday Effects: 45% reduction in routine admissions and 18% increase in emergency visits during holidays.

Weather Correlation: Temperature and rainfall showed a direct link to fluctuations in patient inflow.

Forecasting Accuracy: Prophet and Random Forest models provided the most stable and interpretable predictions.

# Technologies Used

Languages: Python

Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, XGBoost, fbprophet

Tools: Jupyter Notebook, Power BI, Excel

Version Control: Git & GitHub

# Results

Data-driven insights provided to the Communications Team at Cardiff and Vale UHB.

Visual dashboards supported real-time decision-making for resource allocation.

Established an analytical framework that can be extended for other hospital departments.
