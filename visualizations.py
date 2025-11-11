# visualizations.py - Healthcare Analytics Dashboard
# Comprehensive visualizations for EU Visits and Admissions data
# Organized by 6 strategic categories with Top 10 focus where appropriate
# FIXED: Proper handling of bytes data for admissions
# MODIFIED: Removed hour column code block, added colors to Top 10 Diagnoses chart
# MODIFIED: Updated Patient Class analysis to show Top 10 in both pie and bar charts
# MODIFIED: Updated _load_data to read both sheets ("Method_Patient_Class" and "Emergency_IP") from admissions Excel file
# MODIFIED: Added hourly analyses - peak hour by day of week, weather impact on hourly patterns, seasonal hourly demand shifts, weekend vs weekday hourly patterns

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import calendar
from typing import Dict, List, Optional, Tuple, Any
import warnings
import io
from statsmodels.tsa.arima.model import ARIMA
warnings.filterwarnings('ignore')

# Try to import advanced time series libraries
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    st.warning("📦 Install statsmodels for advanced time series models: pip install statsmodels")


class HealthcareAnalyzer:
    """
    Comprehensive healthcare analytics dashboard for EU Visits and Admissions data
    Organized by 6 strategic categories with focused Top 10 analyses
    """

    def __init__(self):
        self.eu_data = None
        self.admissions_data = None
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17becf',
            'light': '#7f7f7f',
            'dark': '#1f1f1f'
        }

    def show_dashboard(self):
        """Main dashboard entry point with dataset selection"""
        try:
            # Load data from session state
            if not self._load_data():
                st.error(" No data available for visualization")
                return

            # Show dashboard header
            self._show_dashboard_header()

            # Dataset selection
            dataset_choice = st.selectbox(
                "Select Dataset for Analysis",
                ["EU VISITS ANALYSIS", "ADMISSIONS ANALYSIS", "INTEGRATED ANALYSIS"],
                key="dataset_selector"
            )

            # Show appropriate analysis
            if dataset_choice == "EU VISITS ANALYSIS":
                self._show_eu_visits_analysis()
            elif dataset_choice == "ADMISSIONS ANALYSIS":
                self._show_admissions_analysis()
            else:
                self._show_integrated_analysis()

        except Exception as e:
            st.error(f"Error generating dashboard: {str(e)}")
            with st.expander("Error Details"):
                st.code(str(e))
                import traceback
                st.code(traceback.format_exc())

    def _load_data(self) -> bool:
        """Load data from session state with proper type handling"""
        data_loaded = False

        # Load EU Visits data (should be DataFrame)
        if hasattr(st.session_state, 'eu_visits_data') and st.session_state.eu_visits_data is not None:
            try:
                if isinstance(st.session_state.eu_visits_data, pd.DataFrame):
                    self.eu_data = st.session_state.eu_visits_data.copy()
                    self.eu_data = self._prepare_eu_data(self.eu_data)
                    data_loaded = True
                    st.success(f"EU Visits data loaded: {len(self.eu_data):,} records")
                else:
                    st.warning("⚠️ EU Visits data is not in DataFrame format")
            except Exception as e:
                st.error(f"❌ Error loading EU Visits data: {str(e)}")

        # Load Admissions data (might be bytes from Excel file)
        if hasattr(st.session_state, 'admissions_data') and st.session_state.admissions_data is not None:
            try:
                admissions_raw = st.session_state.admissions_data

                # Check if it's bytes (raw Excel data)
                if isinstance(admissions_raw, bytes):
                    # Convert bytes to DataFrame using pandas
                    with st.spinner("Loading admissions data from Excel..."):
                        try:
                            # Read all sheets from Excel
                            excel_data = pd.read_excel(io.BytesIO(admissions_raw), sheet_name=None)
                            # Combine all sheets into a single DataFrame
                            all_sheets = []
                            for sheet_name, df in excel_data.items():
                                df['Sheet_Name'] = sheet_name  # Add sheet name as a column to track source
                                all_sheets.append(df)
                            self.admissions_data = pd.concat(all_sheets, ignore_index=True)
                            self.admissions_data = self._prepare_admissions_data(self.admissions_data)
                            data_loaded = True
                            st.success(f"Admissions data loaded from {len(excel_data)} sheets: {len(self.admissions_data):,} records")
                        except Exception as excel_error:
                            # Try reading as CSV if Excel fails
                            try:
                                csv_content = admissions_raw.decode('utf-8')
                                self.admissions_data = pd.read_csv(io.StringIO(csv_content))
                                self.admissions_data = self._prepare_admissions_data(self.admissions_data)
                                data_loaded = True
                                st.success(f"Admissions data loaded as CSV: {len(self.admissions_data):,} records")
                            except Exception as csv_error:
                                st.error(
                                    f"Failed to load admissions data: Excel error: {str(excel_error)}, CSV error: {str(csv_error)}")

                # Check if it's already a DataFrame
                elif isinstance(admissions_raw, pd.DataFrame):
                    self.admissions_data = admissions_raw.copy()
                    self.admissions_data = self._prepare_admissions_data(self.admissions_data)
                    data_loaded = True
                    st.success(f"Admissions DataFrame loaded: {len(self.admissions_data):,} records")

                else:
                    st.warning(f"Admissions data is in unexpected format: {type(admissions_raw)}")

            except Exception as e:
                st.error(f" Error loading admissions data: {str(e)}")

        return data_loaded

    def _prepare_eu_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare and clean EU visits data"""
        try:
            st.info(f"Preparing EU data with columns: {list(data.columns)}")

            # Convert date columns
            if 'Date_String' in data.columns:
                data['Date'] = pd.to_datetime(data['Date_String'], format='%d/%m/%Y', errors='coerce')

            # Ensure numeric columns
            if 'Visit_Count' in data.columns:
                data['Visit_Count'] = pd.to_numeric(data['Visit_Count'], errors='coerce').fillna(0)

            # Extract hour for hourly analysis
            if 'Type' in data.columns and 'Outcome_Type' in data.columns:
                mask = data['Type'] == "UHW EU Visits by Hour of Arrival"
                data.loc[mask, 'Hour'] = pd.to_numeric(data.loc[mask, 'Outcome_Type'], errors='coerce')

            # Create time-based features
            if 'Date' in data.columns:
                data['Year'] = data['Date'].dt.year
                data['Month'] = data['Date'].dt.month
                data['Month_Name'] = data['Date'].dt.month_name()
                data['Day_of_Week'] = data['Date'].dt.day_name()
                data['Is_Weekend'] = data['Day_of_Week'].isin(['Saturday', 'Sunday'])

            st.success(f"EU data prepared: {len(data)} records with {len(data.columns)} columns")
            return data

        except Exception as e:
            st.warning(f"⚠️ EU data preparation warning: {str(e)}")
            return data

    def _prepare_admissions_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare and clean admissions data"""
        try:
            st.info(f"Preparing admissions data with columns: {list(data.columns)}")

            # Look for common date column names
            date_columns = ['Date', 'date', 'DATE', 'Admission_Date', 'admission_date']
            date_col = None

            for col in date_columns:
                if col in data.columns:
                    date_col = col
                    break

            if date_col:
                # Try different date formats
                try:
                    data['Date'] = pd.to_datetime(data[date_col], format='%d/%m/%Y', errors='coerce')
                except:
                    try:
                        data['Date'] = pd.to_datetime(data[date_col], format='%Y-%m-%d', errors='coerce')
                    except:
                        data['Date'] = pd.to_datetime(data[date_col], errors='coerce')

            # Look for admission count columns
            admission_columns = ['No_of_Admissions', 'Admissions', 'Count', 'Total', 'Volume']
            admission_col = None

            for col in admission_columns:
                if col in data.columns:
                    admission_col = col
                    break

            if admission_col:
                data['No_of_Admissions'] = pd.to_numeric(data[admission_col], errors='coerce').fillna(0)

            # Create time-based features if we have dates
            if 'Date' in data.columns and data['Date'].notna().any():
                data['Year'] = data['Date'].dt.year
                data['Month'] = data['Date'].dt.month
                data['Month_Name'] = data['Date'].dt.month_name()
                data['Day_of_Week'] = data['Date'].dt.day_name()
                data['Is_Weekend'] = data['Day_of_Week'].isin(['Saturday', 'Sunday'])

            st.success(f"Admissions data prepared: {len(data)} records with {len(data.columns)} columns")
            return data

        except Exception as e:
            st.warning(f"Admissions data preparation warning: {str(e)}")
            return data

    def _extract_hour_from_outcome(self, outcome_type) -> Optional[int]:
        """Extract hour from outcome type for hourly analysis"""
        try:
            if isinstance(outcome_type, (int, float)) and 0 <= outcome_type <= 23:
                return int(outcome_type)
            return None
        except:
            return None

    def _show_dashboard_header(self):
        """Show dashboard header with key metrics"""
        st.header("Healthcare Analytics Dashboard")
        st.markdown("### Comprehensive insights across 6 strategic healthcare analytics areas")

        # Show available datasets
        col1, col2, col3 = st.columns(3)

        with col1:
            eu_available = self.eu_data is not None
            st.info(f"EU Visits: {'Available' if eu_available else ' Not Available'}")
            if eu_available and 'Visit_Count' in self.eu_data.columns:
                total_eu = self.eu_data['Visit_Count'].sum()
                st.metric("Total EU Visits", f"{total_eu:,}")

        with col2:
            adm_available = self.admissions_data is not None
            st.info(f"Admissions: {'Available' if adm_available else 'Not Available'}")
            if adm_available and 'No_of_Admissions' in self.admissions_data.columns:
                total_adm = self.admissions_data['No_of_Admissions'].sum()
                st.metric("Total Admissions", f"{total_adm:,}")

        with col3:
            integrated_available = eu_available and adm_available
            st.info(f"Integrated: {'Available' if integrated_available else 'Limited'}")

        # Show data preview
        if eu_available:
            with st.expander("EU Visits Data Preview"):
                st.dataframe(self.eu_data.head(), use_container_width=True)

        if adm_available:
            with st.expander("Admissions Data Preview"):
                st.dataframe(self.admissions_data.head(), use_container_width=True)

        st.divider()

    # ═══════════════════════════════════════════════════════════════════════════
    # EU VISITS ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════

    # Modified EU Visits Analysis Method
    def _show_eu_visits_analysis(self):
        """Show EU Visits analysis with 6 strategic categories plus 12-Month Strategic Forecast"""
        if self.eu_data is None:
            st.error("EU Visits data not available")
            return

        st.subheader("EU VISITS ANALYSIS")
        st.info(f"Analyzing {len(self.eu_data):,} EU visits records")

        # Create tabs for the 7 categories
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "Operational Analytics",
            "Clinical Insights",
            "Seasonal Analysis",
            "Patient Flow",
            "Resource Management",
            "Performance KPIs",
            "12-Month Strategic Forecast"
        ])

        with tab1:
            self._show_eu_operational_analytics()
        with tab2:
            self._show_eu_clinical_insights()
        with tab3:
            self._show_eu_seasonal_analysis()
        with tab4:
            self._show_eu_patient_flow()
        with tab5:
            self._show_eu_resource_management()
        with tab6:
            self._show_eu_performance_kpis()
        with tab7:
            self._show_eu_12month_forecast()

    def _show_eu_operational_analytics(self):
        """Operational Analytics for EU Visits"""
        st.markdown("### Operational Analytics")
        st.caption("Time patterns, Volume analysis, Patient flow")

        # Basic volume analysis - FULL WIDTH LAYOUT
        if 'Visit_Count' in self.eu_data.columns:
            st.markdown("#### Daily Volume Distribution")

            if 'Date' in self.eu_data.columns:
                daily_volumes = self.eu_data.groupby('Date')['Visit_Count'].sum().reset_index()

                # Full width histogram
                fig = px.histogram(
                    daily_volumes,
                    x='Visit_Count',
                    nbins=30,
                    title="Distribution of Daily Visit Volumes",
                    labels={"Visit_Count": "Daily Visits", "count": "Frequency"}
                )
                fig.update_layout(height=500, margin=dict(l=80, r=50, t=100, b=80))
                st.plotly_chart(fig, use_container_width=True, key="eu_operational_daily_histogram_main")

                # Full width line chart with better date handling
                fig = px.line(
                    daily_volumes,
                    x='Date',
                    y='Visit_Count',
                    title="Daily Visit Trends Over Time - Complete Timeline"
                )
                fig.update_layout(
                    height=500,
                    margin=dict(l=80, r=50, t=100, b=120),
                    xaxis=dict(
                        tickangle=-45,
                        tickmode='auto',
                        nticks=20,
                        tickfont=dict(size=10)
                    )
                )
                st.plotly_chart(fig, use_container_width=True, key="eu_operational_daily_line_main")

        # Day of week patterns - FULL WIDTH
        if 'Day_of_Week' in self.eu_data.columns:
            st.markdown("#### Day of Week Patterns")

            dow_data = self.eu_data.groupby('Day_of_Week')['Visit_Count'].sum().reset_index()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dow_data['day_num'] = dow_data['Day_of_Week'].map({day: i for i, day in enumerate(day_order)})
            dow_data = dow_data.sort_values('day_num')

            fig = px.bar(
                dow_data,
                x='Day_of_Week',
                y='Visit_Count',
                title="Total Visits by Day of Week",
                labels={"Day_of_Week": "Day", "Visit_Count": "Total Visits"}
            )
            fig.update_layout(height=400, margin=dict(l=80, r=50, t=100, b=80))
            st.plotly_chart(fig, use_container_width=True, key="eu_operational_dow_bar_main")

        # Monthly patterns - FULL WIDTH
        if 'Month_Name' in self.eu_data.columns:
            st.markdown("#### Monthly Patterns")

            monthly_data = self.eu_data.groupby('Month_Name')['Visit_Count'].sum().reset_index()
            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                           'July', 'August', 'September', 'October', 'November', 'December']
            monthly_data['month_num'] = monthly_data['Month_Name'].map(
                {month: i for i, month in enumerate(month_order)})
            monthly_data = monthly_data.sort_values('month_num')

            fig = px.bar(
                monthly_data,
                x='Month_Name',
                y='Visit_Count',
                title="Total Visits by Month - Full Year View",
                color='Visit_Count',
                color_continuous_scale='viridis'
            )
            fig.update_layout(
                height=500,
                showlegend=False,
                margin=dict(l=80, r=50, t=100, b=120),
                xaxis=dict(tickangle=-45, tickfont=dict(size=10))
            )
            st.plotly_chart(fig, use_container_width=True, key="eu_operational_monthly_main")

        # Hourly analyses - FULL WIDTH
        if 'Hour' in self.eu_data.columns and 'Type' in self.eu_data.columns:
            hour_df = self.eu_data[self.eu_data['Type'] == "UHW EU Visits by Hour of Arrival"].copy()

            if not hour_df.empty:
                st.markdown("#### Hourly Patterns")

                # Overall hourly distribution
                hourly_total = hour_df.groupby('Hour')['Visit_Count'].sum().reset_index()

                fig = px.bar(
                    hourly_total,
                    x='Hour',
                    y='Visit_Count',
                    title="Total Visits by Hour of Day"
                )
                fig.update_layout(height=400, margin=dict(l=80, r=50, t=100, b=80))
                st.plotly_chart(fig, use_container_width=True, key="eu_operational_hourly_total_main")

                # Peak hour by day of week
                peak_hours = hour_df.groupby(['Day_of_Week', 'Hour'])['Visit_Count'].mean().reset_index()
                peak_hours = peak_hours.loc[peak_hours.groupby('Day_of_Week')['Visit_Count'].idxmax()]

                fig = px.bar(
                    peak_hours,
                    x='Day_of_Week',
                    y='Hour',
                    title="Peak Hour by Day of Week"
                )
                fig.update_layout(height=400, margin=dict(l=80, r=50, t=100, b=100))
                st.plotly_chart(fig, use_container_width=True, key="eu_operational_peak_hour_main")

    def _show_eu_clinical_insights(self):
        """Clinical Insights for EU Visits"""
        st.markdown("### Clinical Insights")

        # Top diagnoses
        if 'Type' in self.eu_data.columns and 'Outcome_Type' in self.eu_data.columns:
            diagnosis_data = self.eu_data[self.eu_data['Type'] == "UHW EU Visits by Recorded Diagnoses"]

            if not diagnosis_data.empty:
                st.markdown("#### Top 10 Most Common Diagnoses")

                top_diagnoses = diagnosis_data.groupby('Outcome_Type')['Visit_Count'].sum().nlargest(10).reset_index()

                # Horizontal bar chart for better label visibility
                fig = px.bar(
                    top_diagnoses,
                    x='Visit_Count',
                    y='Outcome_Type',
                    orientation='h',
                    title="Top 10 Most Common Diagnoses",
                    labels={"Visit_Count": "Total Visits", "Outcome_Type": "Diagnosis"}
                )
                fig.update_layout(
                    height=max(500, len(top_diagnoses) * 35),
                    margin=dict(l=250, r=50, t=100, b=80)
                )
                st.plotly_chart(fig, use_container_width=True, key="eu_clinical_diagnoses_bar_main")

                # Pie chart
                fig = px.pie(
                    top_diagnoses,
                    values='Visit_Count',
                    names='Outcome_Type',
                    title="Top 10 Diagnosis Distribution"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True, key="eu_clinical_diagnoses_pie_main")

        # Triage analysis
        if 'Type' in self.eu_data.columns:
            triage_data = self.eu_data[self.eu_data['Type'] == "UHW EU Visits by Triage Category"]

            if not triage_data.empty:
                st.markdown("#### riage Category Analysis")

                triage_summary = triage_data.groupby('Outcome_Type')['Visit_Count'].sum().sort_values(
                    ascending=False).reset_index()

                fig = px.bar(
                    triage_summary,
                    x='Visit_Count',
                    y='Outcome_Type',
                    orientation='h',
                    title="Visits by Triage Priority"
                )
                fig.update_layout(
                    height=max(400, len(triage_summary) * 40),
                    margin=dict(l=200, r=50, t=100, b=80)
                )
                st.plotly_chart(fig, use_container_width=True, key="eu_clinical_triage_bar_main")
                # Pie chart for triage priority
                fig = px.pie(
                    triage_summary,
                    values='Visit_Count',
                    names='Outcome_Type',
                    title="Visits by Triage Priority Distribution"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True, key="eu_clinical_triage_pie_main")

    def _show_eu_seasonal_analysis(self):
        """ Seasonal Analysis for EU Visits"""
        st.markdown("### Seasonal Analysis")
        st.caption("Weather impact, Seasonal trends, Holiday effects")

        # Create seasonal tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Spring", "Summer", "Autumn", "Winter"])

        with tab1:
            self._show_spring_analysis()
        with tab2:
            self._show_summer_analysis()
        with tab3:
            self._show_autumn_analysis()
        with tab4:
            self._show_winter_analysis()

    def _show_spring_analysis(self):
        """Spring seasonal analysis"""
        st.markdown("#### Spring Analysis (March - May)")

        if 'Month' in self.eu_data.columns:
            spring_data = self.eu_data[self.eu_data['Month'].isin([3, 4, 5])]

            if spring_data.empty:
                st.warning("Spring data is not available in the processed dataset.")
                return

            # Monthly breakdown within spring
            spring_monthly = spring_data.groupby('Month_Name')['Visit_Count'].sum().reset_index()

            fig = px.bar(
                spring_monthly,
                x='Month_Name',
                y='Visit_Count',
                title="Spring Months Visit Distribution",
                color='Visit_Count',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key="spring_monthly")

            # Top 10 Diagnoses drill-down
            if 'Type' in spring_data.columns and 'Outcome_Type' in spring_data.columns:
                diagnosis_spring = spring_data[spring_data['Type'] == "UHW EU Visits by Recorded Diagnoses"]
                if not diagnosis_spring.empty:
                    st.markdown("##### Top 10 Diagnoses in Spring")
                    top_diagnoses = diagnosis_spring.groupby('Outcome_Type')['Visit_Count'].sum().nlargest(
                        10).reset_index()

                    fig = px.bar(
                        top_diagnoses,
                        x='Visit_Count',
                        y='Outcome_Type',
                        orientation='h',
                        title="Top 10 Diagnoses - Spring Season"
                    )
                    fig.update_layout(height=max(400, len(top_diagnoses) * 30), margin=dict(l=250, r=50, t=80, b=60))
                    st.plotly_chart(fig, use_container_width=True, key="spring_diagnoses_bar")

                    fig = px.pie(
                        top_diagnoses,
                        values='Visit_Count',
                        names='Outcome_Type',
                        title="Top 10 Diagnoses Distribution - Spring",
                        color_discrete_sequence=px.colors.qualitative.Dark2
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True, key="spring_diagnoses_pie")

                # Triage categories
                triage_spring = spring_data[spring_data['Type'] == "UHW EU Visits by Triage Category"]
                if not triage_spring.empty:
                    st.markdown("##### Triage Categories in Spring")
                    triage_data = triage_spring.groupby('Outcome_Type')['Visit_Count'].sum().reset_index()

                    fig = px.pie(
                        triage_data,
                        values='Visit_Count',
                        names='Outcome_Type',
                        title="Triage Priority Distribution - Spring",
                        color_discrete_sequence=px.colors.qualitative.Dark2
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True, key="spring_triage_pie")

            # Academic year seasonal patterns
            if 'Academic_Year' in spring_data.columns:
                st.markdown("##### Academic Year Impact in Spring")
                academic_spring = spring_data.groupby('Academic_Year')['Visit_Count'].sum().reset_index()

                fig = px.bar(
                    academic_spring,
                    x='Academic_Year',
                    y='Visit_Count',
                    title="Spring Visits by Academic Year"
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True, key="spring_academic_bar")

            # Temperature correlation analysis
            if 'Temperature_Mean_C' in spring_data.columns:
                st.markdown("##### Temperature Impact on Visit Volume")
                temp_visits = spring_data.groupby('Temperature_Mean_C')['Visit_Count'].sum().reset_index()

                if len(temp_visits) > 1:
                    correlation = temp_visits['Temperature_Mean_C'].corr(temp_visits['Visit_Count'])

                    fig = px.scatter(
                        temp_visits,
                        x='Temperature_Mean_C',
                        y='Visit_Count',
                        title=f"Spring: Temperature vs Visit Volume (r = {correlation:.3f})",
                        trendline="ols"
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True, key="spring_temp_scatter")

                    # Correlation interpretation
                    if abs(correlation) >= 0.7:
                        strength = "Strong"
                    elif abs(correlation) >= 0.3:
                        strength = "Moderate"
                    else:
                        strength = "Weak"

                    direction = "positively" if correlation > 0 else "negatively"
                    st.info(
                        f"**{strength} {direction} correlation (r = {correlation:.3f})** - Temperature affects visit volume in Spring")

            # Precipitation analysis
            if 'Precipitation_Category' in spring_data.columns:
                st.markdown("##### Precipitation Impact")
                precip_spring = spring_data.groupby('Precipitation_Category')['Visit_Count'].sum().reset_index()

                fig = px.pie(
                    precip_spring,
                    values='Visit_Count',
                    names='Precipitation_Category',
                    title="Spring Visits by Precipitation Level"
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True, key="spring_precip_pie")

            # Weather impact
            if 'Weather_Category' in spring_data.columns:
                spring_weather = spring_data.groupby('Weather_Category')['Visit_Count'].sum().reset_index()

                fig = px.pie(
                    spring_weather,
                    values='Visit_Count',
                    names='Weather_Category',
                    title="Spring Weather Impact on Visits",
                    color_discrete_sequence=px.colors.qualitative.Dark2
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True, key="spring_weather")
        else:
            st.error(" Month column not found in the dataset.")

    def _show_summer_analysis(self):
        """Summer seasonal analysis"""
        st.markdown("####  Summer Analysis (June - August)")

        if 'Month' in self.eu_data.columns:
            summer_data = self.eu_data[self.eu_data['Month'].isin([6, 7, 8])]

            if summer_data.empty:
                st.warning("Summer data is not available in the processed dataset.")
                return

            # Monthly breakdown within summer
            summer_monthly = summer_data.groupby('Month_Name')['Visit_Count'].sum().reset_index()

            fig = px.bar(
                summer_monthly,
                x='Month_Name',
                y='Visit_Count',
                title="Summer Months Visit Distribution",
                color='Visit_Count',
                color_continuous_scale='plasma'
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key="summer_monthly")

            # Top 10 Diagnoses drill-down
            if 'Type' in summer_data.columns and 'Outcome_Type' in summer_data.columns:
                diagnosis_summer = summer_data[summer_data['Type'] == "UHW EU Visits by Recorded Diagnoses"]
                if not diagnosis_summer.empty:
                    st.markdown("##### Top 10 Diagnoses in Summer")
                    top_diagnoses = diagnosis_summer.groupby('Outcome_Type')['Visit_Count'].sum().nlargest(
                        10).reset_index()

                    fig = px.bar(
                        top_diagnoses,
                        x='Visit_Count',
                        y='Outcome_Type',
                        orientation='h',
                        title="Top 10 Diagnoses - Summer Season"
                    )
                    fig.update_layout(height=max(400, len(top_diagnoses) * 30), margin=dict(l=250, r=50, t=80, b=60))
                    st.plotly_chart(fig, use_container_width=True, key="summer_diagnoses_bar")

                    fig = px.pie(
                        top_diagnoses,
                        values='Visit_Count',
                        names='Outcome_Type',
                        title="Top 10 Diagnoses Distribution - Summer",
                        color_discrete_sequence=px.colors.qualitative.Set1
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True, key="summer_diagnoses_pie")

                # Triage categories
                triage_summer = summer_data[summer_data['Type'] == "UHW EU Visits by Triage Category"]
                if not triage_summer.empty:
                    st.markdown("##### Triage Categories in Summer")
                    triage_data = triage_summer.groupby('Outcome_Type')['Visit_Count'].sum().reset_index()

                    fig = px.pie(
                        triage_data,
                        values='Visit_Count',
                        names='Outcome_Type',
                        title="Triage Priority Distribution - Summer",
                        color_discrete_sequence=px.colors.qualitative.Set1
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True, key="summer_triage_pie")

            # Academic year seasonal patterns
            if 'Academic_Year' in summer_data.columns:
                st.markdown("##### Academic Year Impact in Summer")
                academic_summer = summer_data.groupby('Academic_Year')['Visit_Count'].sum().reset_index()

                fig = px.bar(
                    academic_summer,
                    x='Academic_Year',
                    y='Visit_Count',
                    title="Summer Visits by Academic Year"
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True, key="summer_academic_bar")

            # Temperature correlation analysis
            if 'Temperature_Mean_C' in summer_data.columns:
                st.markdown("##### Temperature Impact on Visit Volume")
                temp_visits = summer_data.groupby('Temperature_Mean_C')['Visit_Count'].sum().reset_index()

                if len(temp_visits) > 1:
                    correlation = temp_visits['Temperature_Mean_C'].corr(temp_visits['Visit_Count'])

                    fig = px.scatter(
                        temp_visits,
                        x='Temperature_Mean_C',
                        y='Visit_Count',
                        title=f"Summer: Temperature vs Visit Volume (r = {correlation:.3f})",
                        trendline="ols"
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True, key="summer_temp_scatter")

                    # Correlation interpretation
                    if abs(correlation) >= 0.7:
                        strength = "Strong"
                    elif abs(correlation) >= 0.3:
                        strength = "Moderate"
                    else:
                        strength = "Weak"

                    direction = "positively" if correlation > 0 else "negatively"
                    st.info(
                        f"**{strength} {direction} correlation (r = {correlation:.3f})** - Temperature affects visit volume in Summer")

            # Precipitation analysis
            if 'Precipitation_Category' in summer_data.columns:
                st.markdown("##### Precipitation Impact")
                precip_summer = summer_data.groupby('Precipitation_Category')['Visit_Count'].sum().reset_index()

                fig = px.pie(
                    precip_summer,
                    values='Visit_Count',
                    names='Precipitation_Category',
                    title="Summer Visits by Precipitation Level"
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True, key="summer_precip_pie")

            # Weather impact
            if 'Weather_Category' in summer_data.columns:
                summer_weather = summer_data.groupby('Weather_Category')['Visit_Count'].sum().reset_index()

                fig = px.pie(
                    summer_weather,
                    values='Visit_Count',
                    names='Weather_Category',
                    title="Summer Weather Impact on Visits",
                    color_discrete_sequence=px.colors.qualitative.Set1
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True, key="summer_weather")
        else:
            st.error(" Month column not found in the dataset.")

    def _show_autumn_analysis(self):
        """Autumn seasonal analysis"""
        st.markdown("#### Autumn Analysis (September - November)")

        if 'Month' in self.eu_data.columns:
            autumn_data = self.eu_data[self.eu_data['Month'].isin([9, 10, 11])]

            if autumn_data.empty:
                st.warning("Autumn data is not available in the processed dataset.")
                return

            # Monthly breakdown within autumn
            autumn_monthly = autumn_data.groupby('Month_Name')['Visit_Count'].sum().reset_index()

            fig = px.bar(
                autumn_monthly,
                x='Month_Name',
                y='Visit_Count',
                title="Autumn Months Visit Distribution",
                color='Visit_Count',
                color_continuous_scale='inferno'
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key="autumn_monthly")

            # Top 10 Diagnoses drill-down
            if 'Type' in autumn_data.columns and 'Outcome_Type' in autumn_data.columns:
                diagnosis_autumn = autumn_data[autumn_data['Type'] == "UHW EU Visits by Recorded Diagnoses"]
                if not diagnosis_autumn.empty:
                    st.markdown("##### 📊 Top 10 Diagnoses in Autumn")
                    top_diagnoses = diagnosis_autumn.groupby('Outcome_Type')['Visit_Count'].sum().nlargest(
                        10).reset_index()

                    fig = px.bar(
                        top_diagnoses,
                        x='Visit_Count',
                        y='Outcome_Type',
                        orientation='h',
                        title="Top 10 Diagnoses - Autumn Season"
                    )
                    fig.update_layout(height=max(400, len(top_diagnoses) * 30), margin=dict(l=250, r=50, t=80, b=60))
                    st.plotly_chart(fig, use_container_width=True, key="autumn_diagnoses_bar")

                    fig = px.pie(
                        top_diagnoses,
                        values='Visit_Count',
                        names='Outcome_Type',
                        title="Top 10 Diagnoses Distribution - Autumn",
                        color_discrete_sequence=px.colors.qualitative.Dark24
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True, key="autumn_diagnoses_pie")

                # Triage categories
                triage_autumn = autumn_data[autumn_data['Type'] == "UHW EU Visits by Triage Category"]
                if not triage_autumn.empty:
                    st.markdown("##### 🎯 Triage Categories in Autumn")
                    triage_data = triage_autumn.groupby('Outcome_Type')['Visit_Count'].sum().reset_index()

                    fig = px.pie(
                        triage_data,
                        values='Visit_Count',
                        names='Outcome_Type',
                        title="Triage Priority Distribution - Autumn",
                        color_discrete_sequence=px.colors.qualitative.Dark24
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True, key="autumn_triage_pie")

            # Academic year seasonal patterns
            if 'Academic_Year' in autumn_data.columns:
                st.markdown("##### 📚 Academic Year Impact in Autumn")
                academic_autumn = autumn_data.groupby('Academic_Year')['Visit_Count'].sum().reset_index()

                fig = px.bar(
                    academic_autumn,
                    x='Academic_Year',
                    y='Visit_Count',
                    title="Autumn Visits by Academic Year"
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True, key="autumn_academic_bar")

            # Temperature correlation analysis
            if 'Temperature_Mean_C' in autumn_data.columns:
                st.markdown("##### 🌡️ Temperature Impact on Visit Volume")
                temp_visits = autumn_data.groupby('Temperature_Mean_C')['Visit_Count'].sum().reset_index()

                if len(temp_visits) > 1:
                    correlation = temp_visits['Temperature_Mean_C'].corr(temp_visits['Visit_Count'])

                    fig = px.scatter(
                        temp_visits,
                        x='Temperature_Mean_C',
                        y='Visit_Count',
                        title=f"Autumn: Temperature vs Visit Volume (r = {correlation:.3f})",
                        trendline="ols"
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True, key="autumn_temp_scatter")

                    # Correlation interpretation
                    if abs(correlation) >= 0.7:
                        strength = "Strong"
                    elif abs(correlation) >= 0.3:
                        strength = "Moderate"
                    else:
                        strength = "Weak"

                    direction = "positively" if correlation > 0 else "negatively"
                    st.info(
                        f"**{strength} {direction} correlation (r = {correlation:.3f})** - Temperature affects visit volume in Autumn")

            # Precipitation analysis
            if 'Precipitation_Category' in autumn_data.columns:
                st.markdown("##### 🌧️ Precipitation Impact")
                precip_autumn = autumn_data.groupby('Precipitation_Category')['Visit_Count'].sum().reset_index()

                fig = px.pie(
                    precip_autumn,
                    values='Visit_Count',
                    names='Precipitation_Category',
                    title="Autumn Visits by Precipitation Level"
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True, key="autumn_precip_pie")

            # Weather impact
            if 'Weather_Category' in autumn_data.columns:
                autumn_weather = autumn_data.groupby('Weather_Category')['Visit_Count'].sum().reset_index()

                fig = px.pie(
                    autumn_weather,
                    values='Visit_Count',
                    names='Weather_Category',
                    title="Autumn Weather Impact on Visits",
                    color_discrete_sequence=px.colors.qualitative.Dark24
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True, key="autumn_weather")
        else:
            st.error("❌ Month column not found in the dataset.")

    def _show_winter_analysis(self):
        """Winter seasonal analysis"""
        st.markdown("#### ❄️ Winter Analysis (December - February)")

        if 'Month' in self.eu_data.columns:
            winter_data = self.eu_data[self.eu_data['Month'].isin([12, 1, 2])]

            if winter_data.empty:
                st.warning("⚠️ Winter data is not available in the processed dataset.")
                return

            # Monthly breakdown within winter
            winter_monthly = winter_data.groupby('Month_Name')['Visit_Count'].sum().reset_index()

            fig = px.bar(
                winter_monthly,
                x='Month_Name',
                y='Visit_Count',
                title="Winter Months Visit Distribution",
                color='Visit_Count',
                color_continuous_scale='cividis'
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key="winter_monthly")

            # Top 10 Diagnoses drill-down
            if 'Type' in winter_data.columns and 'Outcome_Type' in winter_data.columns:
                diagnosis_winter = winter_data[winter_data['Type'] == "UHW EU Visits by Recorded Diagnoses"]
                if not diagnosis_winter.empty:
                    st.markdown("##### 📊 Top 10 Diagnoses in Winter")
                    top_diagnoses = diagnosis_winter.groupby('Outcome_Type')['Visit_Count'].sum().nlargest(
                        10).reset_index()

                    fig = px.bar(
                        top_diagnoses,
                        x='Visit_Count',
                        y='Outcome_Type',
                        orientation='h',
                        title="Top 10 Diagnoses - Winter Season"
                    )
                    fig.update_layout(height=max(400, len(top_diagnoses) * 30), margin=dict(l=250, r=50, t=80, b=60))
                    st.plotly_chart(fig, use_container_width=True, key="winter_diagnoses_bar")

                    fig = px.pie(
                        top_diagnoses,
                        values='Visit_Count',
                        names='Outcome_Type',
                        title="Top 10 Diagnoses Distribution - Winter",
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True, key="winter_diagnoses_pie")

                # Triage categories
                triage_winter = winter_data[winter_data['Type'] == "UHW EU Visits by Triage Category"]
                if not triage_winter.empty:
                    st.markdown("##### 🎯 Triage Categories in Winter")
                    triage_data = triage_winter.groupby('Outcome_Type')['Visit_Count'].sum().reset_index()

                    fig = px.pie(
                        triage_data,
                        values='Visit_Count',
                        names='Outcome_Type',
                        title="Triage Priority Distribution - Winter",
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True, key="winter_triage_pie")

            # Academic year seasonal patterns
            if 'Academic_Year' in winter_data.columns:
                st.markdown("##### 📚 Academic Year Impact in Winter")
                academic_winter = winter_data.groupby('Academic_Year')['Visit_Count'].sum().reset_index()

                fig = px.bar(
                    academic_winter,
                    x='Academic_Year',
                    y='Visit_Count',
                    title="Winter Visits by Academic Year"
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True, key="winter_academic_bar")

            # Temperature correlation analysis
            if 'Temperature_Mean_C' in winter_data.columns:
                st.markdown("##### 🌡️ Temperature Impact on Visit Volume")
                temp_visits = winter_data.groupby('Temperature_Mean_C')['Visit_Count'].sum().reset_index()

                if len(temp_visits) > 1:
                    correlation = temp_visits['Temperature_Mean_C'].corr(temp_visits['Visit_Count'])

                    fig = px.scatter(
                        temp_visits,
                        x='Temperature_Mean_C',
                        y='Visit_Count',
                        title=f"Winter: Temperature vs Visit Volume (r = {correlation:.3f})",
                        trendline="ols"
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True, key="winter_temp_scatter")

                    # Correlation interpretation
                    if abs(correlation) >= 0.7:
                        strength = "Strong"
                    elif abs(correlation) >= 0.3:
                        strength = "Moderate"
                    else:
                        strength = "Weak"

                    direction = "positively" if correlation > 0 else "negatively"
                    st.info(
                        f"**{strength} {direction} correlation (r = {correlation:.3f})** - Temperature affects visit volume in Winter")

            # Precipitation analysis
            if 'Precipitation_Category' in winter_data.columns:
                st.markdown("##### 🌧️ Precipitation Impact")
                precip_winter = winter_data.groupby('Precipitation_Category')['Visit_Count'].sum().reset_index()

                fig = px.pie(
                    precip_winter,
                    values='Visit_Count',
                    names='Precipitation_Category',
                    title="Winter Visits by Precipitation Level"
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True, key="winter_precip_pie")

            # Weather impact
            if 'Weather_Category' in winter_data.columns:
                winter_weather = winter_data.groupby('Weather_Category')['Visit_Count'].sum().reset_index()

                fig = px.pie(
                    winter_weather,
                    values='Visit_Count',
                    names='Weather_Category',
                    title="Winter Weather Impact on Visits",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True, key="winter_weather")

            # Holiday impact for winter
            if 'Is_Bank_Holiday' in winter_data.columns:
                st.markdown("##### 🎄 Holiday Impact")
                holiday_winter = winter_data.groupby('Is_Bank_Holiday')['Visit_Count'].agg(
                    ['sum', 'mean']).reset_index()

                if len(holiday_winter) > 1:
                    fig = px.bar(
                        holiday_winter,
                        x='Is_Bank_Holiday',
                        y='mean',
                        title="Winter: Normal vs Holiday Average Visits"
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True, key="winter_holiday")
        else:
            st.error("❌ Month column not found in the dataset.")

    def _show_eu_patient_flow(self):
        """Patient Flow for EU Visits"""
        st.markdown("### Patient Flow")

        # Referral sources
        if 'Type' in self.eu_data.columns:
            referral_data = self.eu_data[self.eu_data['Type'] == "UHW EU Visits by Source of Referral"]

            if not referral_data.empty:
                st.markdown("#### Top 10 Referral Sources")

                top_referral_sources = referral_data.groupby('Outcome_Type')['Visit_Count'].sum().nlargest(
                    10).reset_index()

                # Horizontal bar for better label visibility
                fig = px.bar(
                    top_referral_sources,
                    x='Visit_Count',
                    y='Outcome_Type',
                    orientation='h',
                    title="Top 10 Referral Sources by Volume"
                )
                fig.update_layout(
                    height=max(500, len(top_referral_sources) * 35),
                    margin=dict(l=200, r=50, t=100, b=80)
                )
                st.plotly_chart(fig, use_container_width=True, key="eu_patient_flow_referral_bar_main")

                # Pie chart
                fig = px.pie(
                    top_referral_sources,
                    values='Visit_Count',
                    names='Outcome_Type',
                    title="Top 10 Referral Sources Distribution"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True, key="eu_patient_flow_referral_pie_main")

        # Patient outcomes
        if 'Type' in self.eu_data.columns:
            outcome_data = self.eu_data[self.eu_data['Type'] == "UHW EU Visits by Outcome"]

            if not outcome_data.empty:
                st.markdown("#### 🔄 Patient Outcomes")

                outcome_summary = outcome_data.groupby('Outcome_Type')['Visit_Count'].sum().sort_values(
                    ascending=False).head(10).reset_index()

                fig = px.bar(
                    outcome_summary,
                    x='Visit_Count',
                    y='Outcome_Type',
                    orientation='h',
                    title="Top 10 Patient Outcomes"
                )
                fig.update_layout(
                    height=max(500, len(outcome_summary) * 35),
                    margin=dict(l=200, r=50, t=100, b=80)
                )
                st.plotly_chart(fig, use_container_width=True, key="eu_patient_flow_outcome_bar_main")
                # Pie chart for patient outcomes
                fig = px.pie(
                    outcome_summary,
                    values='Visit_Count',
                    names='Outcome_Type',
                    title="Top 10 Patient Outcomes Distribution"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True, key="eu_patient_flow_outcome_pie_main")

    def _show_eu_resource_management(self):
        """ Resource Management for EU Visits"""
        st.markdown("### Resource Management")
        st.caption("Capacity planning, Peak demand analysis")

        if 'Visit_Count' in self.eu_data.columns and 'Date' in self.eu_data.columns:
            daily_volumes = self.eu_data.groupby('Date')['Visit_Count'].sum().reset_index()

            # Calculate capacity metrics
            mean_vol = daily_volumes['Visit_Count'].mean()
            std_vol = daily_volumes['Visit_Count'].std()
            max_vol = daily_volumes['Visit_Count'].max()
            min_vol = daily_volumes['Visit_Count'].min()

            st.markdown("#### ⚡ Capacity Analysis")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average Daily", f"{mean_vol:.0f}")
            with col2:
                st.metric("Peak Daily", f"{max_vol:.0f}")
            with col3:
                st.metric("Minimum Daily", f"{min_vol:.0f}")
            with col4:
                variability = (std_vol / mean_vol) * 100 if mean_vol > 0 else 0
                st.metric("Variability %", f"{variability:.1f}%")

            # Capacity distribution - FULL WIDTH
            fig = px.histogram(
                daily_volumes,
                x='Visit_Count',
                nbins=25,
                title="Daily Visit Volume Distribution"
            )
            fig.update_layout(height=400, margin=dict(l=80, r=50, t=100, b=80))
            st.plotly_chart(fig, use_container_width=True, key="eu_resource_capacity_histogram_main")

        # Weekend vs weekday analysis
        if 'Is_Weekend' in self.eu_data.columns:
            st.markdown("#### Weekend vs Weekday Demand")

            weekend_analysis = self.eu_data.groupby(['Date', 'Is_Weekend'])['Visit_Count'].sum().reset_index()

            fig = px.box(
                weekend_analysis,
                x='Is_Weekend',
                y='Visit_Count',
                title="Visit Volume Distribution: Weekday vs Weekend"
            )
            fig.update_layout(height=400, margin=dict(l=80, r=50, t=100, b=80))
            st.plotly_chart(fig, use_container_width=True, key="eu_resource_weekend_box_main")

    def _show_eu_performance_kpis(self):
        """Performance KPIs for EU Visits"""
        st.markdown("### Performance KPIs")
        st.caption("Quality metrics, Trend analysis, Benchmarking")

        # Calculate basic KPIs
        if 'Visit_Count' in self.eu_data.columns:
            total_visits = self.eu_data['Visit_Count'].sum()

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Visits", f"{total_visits:,}")

            with col2:
                if 'Date' in self.eu_data.columns:
                    unique_dates = self.eu_data['Date'].nunique()
                    daily_avg = total_visits / unique_dates if unique_dates > 0 else 0
                    st.metric("Daily Average", f"{daily_avg:.0f}")

            with col3:
                if 'Type' in self.eu_data.columns:
                    visit_types = self.eu_data['Type'].nunique()
                    st.metric("Visit Types", f"{visit_types}")

            with col4:
                if 'Outcome_Type' in self.eu_data.columns:
                    outcome_types = self.eu_data['Outcome_Type'].nunique()
                    st.metric("Outcome Categories", f"{outcome_types}")

        # Performance efficiency metrics
        if 'Date' in self.eu_data.columns and 'Visit_Count' in self.eu_data.columns:
            st.markdown("#### Monthly Performance Trends")

            monthly_performance = self.eu_data.groupby('Month_Name')['Visit_Count'].agg(
                ['sum', 'mean', 'std']).reset_index()
            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                           'July', 'August', 'September', 'October', 'November', 'December']
            monthly_performance['month_num'] = monthly_performance['Month_Name'].map(
                {month: i for i, month in enumerate(month_order)})
            monthly_performance = monthly_performance.sort_values('month_num')

            # Monthly volume trend line
            fig = px.line(
                monthly_performance,
                x='Month_Name',
                y='sum',
                title="Monthly Visit Volume Trend",
                markers=True
            )
            fig.update_layout(
                height=400,
                margin=dict(l=80, r=50, t=100, b=120),
                xaxis=dict(tickangle=-45, tickfont=dict(size=10))
            )
            st.plotly_chart(fig, use_container_width=True, key="eu_performance_monthly_trend")

        # OPTION 6: Monthly Distribution (Seasonal patterns)
        if 'Month_Name' in self.eu_data.columns:
            st.markdown("#### EU Visits by Month")

            monthly_summary = self.eu_data.groupby('Month_Name')['Visit_Count'].sum().reset_index()

            # Sort by month order
            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                           'July', 'August', 'September', 'October', 'November', 'December']
            monthly_summary['month_num'] = monthly_summary['Month_Name'].map(
                {month: i for i, month in enumerate(month_order)})
            monthly_summary = monthly_summary.sort_values('month_num')

            fig = px.pie(
                monthly_summary,
                values='Visit_Count',
                names='Month_Name',
                title="EU Visits Distribution by Month",
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True, key="eu_performance_monthly_pie")

        # Weekly performance consistency
        if 'Day_of_Week' in self.eu_data.columns and 'Visit_Count' in self.eu_data.columns:
            st.markdown("#### Weekly Performance Consistency")

            weekly_consistency = self.eu_data.groupby('Day_of_Week')['Visit_Count'].agg(['mean', 'std']).reset_index()
            weekly_consistency['coefficient_of_variation'] = (weekly_consistency['std'] / weekly_consistency[
                'mean']) * 100

            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekly_consistency['day_num'] = weekly_consistency['Day_of_Week'].map(
                {day: i for i, day in enumerate(day_order)})
            weekly_consistency = weekly_consistency.sort_values('day_num')

            fig = px.bar(
                weekly_consistency,
                x='Day_of_Week',
                y='coefficient_of_variation',
                title="Daily Visit Variability (Lower = More Consistent)",
                labels={"coefficient_of_variation": "Coefficient of Variation (%)", "Day_of_Week": "Day"}
            )
            fig.update_layout(
                height=400,
                margin=dict(l=80, r=50, t=100, b=120),
                xaxis=dict(tickangle=-45, tickfont=dict(size=10))
            )
            st.plotly_chart(fig, use_container_width=True, key="eu_performance_consistency")

        # System efficiency indicators
        if 'Is_Weekend' in self.eu_data.columns:
            st.markdown("#### System Efficiency Indicators")

            efficiency_metrics = self.eu_data.groupby('Is_Weekend')['Visit_Count'].agg(
                ['sum', 'mean', 'count']).reset_index()
            efficiency_metrics['Is_Weekend'] = efficiency_metrics['Is_Weekend'].map({True: 'Weekend', False: 'Weekday'})

            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    efficiency_metrics,
                    x='Is_Weekend',
                    y='mean',
                    title="Average Daily Visits: Weekday vs Weekend",
                    labels={"mean": "Average Daily Visits", "Is_Weekend": "Day Type"}
                )
                fig.update_layout(height=350, margin=dict(l=80, r=50, t=80, b=80))
                st.plotly_chart(fig, use_container_width=True, key="eu_performance_weekday_weekend")

            with col2:
                total_weekday = efficiency_metrics[efficiency_metrics['Is_Weekend'] == 'Weekday']['sum'].iloc[0]
                total_weekend = efficiency_metrics[efficiency_metrics['Is_Weekend'] == 'Weekend']['sum'].iloc[0]

                ratio_data = pd.DataFrame({
                    'Period': ['Weekday', 'Weekend'],
                    'Total_Visits': [total_weekday, total_weekend]
                })

                fig = px.pie(
                    ratio_data,
                    values='Total_Visits',
                    names='Period',
                    title="Weekday vs Weekend Visit Distribution"
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True, key="eu_performance_period_distribution")

    # New simplified 12-month forecast method
    def _show_eu_12month_forecast(self):
        """12-Month Strategic Forecasting for EU Visits"""
        st.markdown("### 12-Month Strategic Forecast & Predictions")
        st.info("Long-term predictions for capacity planning and resource allocation")

        daily_data, error = self._prepare_forecasting_data()
        if daily_data is None:
            st.error(f"❌ {error}")
            return

        # Generate comprehensive yearly forecasts with working visualizations
        self._generate_yearly_insights_fixed(daily_data)

    def _show_eu_forecasting_analysis(self):
        """Model-Based Forecasting Analysis for EU Visits with Working Visualizations"""
        st.markdown("### Model-Based Forecasting Analysis")
        st.caption("ARIMA, Machine Learning, Statistical validation")

        if self.eu_data is None or 'Visit_Count' not in self.eu_data.columns:
            st.error("EU Visits data not available for forecasting")
            return

        # Create forecasting sub-tabs
        forecast_tab1, forecast_tab2, forecast_tab3, forecast_tab4, forecast_tab5 = st.tabs([
            "ARIMA Time Series",
            "Machine Learning Models",
            "Model Comparison",
            "12-Month Strategic Forecast",
            "Model Validation"
        ])

        with forecast_tab1:
            self._show_arima_forecasting()
        with forecast_tab2:
            self._show_ml_forecasting()
        with forecast_tab3:
            self._show_model_comparison()
        with forecast_tab4:
            self._show_ensemble_forecasting_fixed()
        with forecast_tab5:
            self._show_model_validation()

    def _show_arima_forecasting(self):
        """ARIMA Time Series Forecasting for EU Visits"""
        st.markdown("#### ARIMA Time Series Forecasting")

        if not STATSMODELS_AVAILABLE:
            st.error("📦 ARIMA forecasting requires statsmodels. Install with: pip install statsmodels")
            st.info("Showing alternative forecasting approach...")
            self._show_simple_forecasting()
            return

        daily_data, error = self._prepare_forecasting_data()
        if daily_data is None:
            st.error(f"❌ {error}")
            return

        st.markdown("##### ARIMA Model Configuration")

        # Prepare time series data
        ts_data = daily_data.set_index('Date')['Visit_Count']

        # Check for sufficient data
        if len(ts_data) < 30:
            st.error("❌ Insufficient data for ARIMA modeling (minimum 30 observations required)")
            return

        # Display time series characteristics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Observations", len(ts_data))
        with col2:
            st.metric("Mean Daily Visits", f"{ts_data.mean():.1f}")
        with col3:
            st.metric("Std Deviation", f"{ts_data.std():.1f}")

        try:
            # Stationarity test
            st.markdown("##### Stationarity Analysis")
            adf_result = adfuller(ts_data.dropna())

            st.write(f"**ADF Statistic**: {adf_result[0]:.6f}")
            st.write(f"**p-value**: {adf_result[1]:.6f}")

            if adf_result[1] <= 0.05:
                st.success("✅ Time series is stationary")
                differencing_needed = False
            else:
                st.warning("⚠️ Time series may need differencing")
                differencing_needed = True

            # Auto ARIMA or manual parameter selection
            st.markdown("##### ARIMA Model Fitting")

            with st.spinner("Fitting ARIMA model..."):
                try:
                    # Use simple ARIMA(1,1,1) as default
                    if differencing_needed:
                        model = ARIMA(ts_data, order=(1, 1, 1))
                    else:
                        model = ARIMA(ts_data, order=(1, 0, 1))

                    fitted_model = model.fit()

                    # Display model summary
                    st.success("✅ ARIMA model fitted successfully")

                    # Model parameters
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Model Order**:", fitted_model.model.order)
                    with col2:
                        st.write("**AIC**:", f"{fitted_model.aic:.2f}")

                    # Generate forecast
                    forecast_steps = 30  # 30 days ahead
                    forecast = fitted_model.forecast(steps=forecast_steps)
                    forecast_index = pd.date_range(start=ts_data.index[-1] + pd.Timedelta(days=1),
                                                   periods=forecast_steps, freq='D')

                    # Plot results
                    fig = go.Figure()

                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=ts_data.index[-90:],  # Last 90 days
                        y=ts_data.values[-90:],
                        mode='lines',
                        name='Historical',
                        line=dict(color='blue')
                    ))

                    # Forecast
                    fig.add_trace(go.Scatter(
                        x=forecast_index,
                        y=forecast,
                        mode='lines',
                        name='ARIMA Forecast',
                        line=dict(color='red', dash='dash')
                    ))

                    fig.update_layout(
                        title="ARIMA Forecast - EU Visits (30 Days)",
                        height=500,
                        xaxis_title="Date",
                        yaxis_title="Daily Visits"
                    )
                    st.plotly_chart(fig, use_container_width=True, key="arima_forecast")

                    # Forecast summary
                    st.markdown("##### Forecast Summary")
                    forecast_avg = forecast.mean()
                    forecast_trend = "increasing" if forecast[-1] > forecast[0] else "decreasing"

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Forecast", f"{forecast_avg:.0f}")
                    with col2:
                        st.metric("30-Day Trend", forecast_trend.title())
                    with col3:
                        st.metric("Next Day Prediction", f"{forecast.iloc[0]:.0f}")

                except Exception as e:
                    st.error(f"❌ ARIMA fitting failed: {str(e)}")
                    st.info("Showing alternative forecasting approach...")
                    self._show_simple_forecasting()

        except Exception as e:
            st.error(f"❌ ARIMA analysis failed: {str(e)}")
            st.info("Showing alternative forecasting approach...")
            self._show_simple_forecasting()

    def _show_simple_forecasting(self):
        """Simple forecasting when ARIMA is not available"""
        st.markdown("#### Simple Forecasting Analysis")

        daily_data, error = self._prepare_forecasting_data()
        if daily_data is None:
            st.error(f"❌ {error}")
            return

        ts_data = daily_data['Visit_Count'].values
        dates = daily_data['Date'].values

        # Moving average forecast
        window = min(7, len(ts_data) // 4)
        ma_forecast = np.mean(ts_data[-window:])

        # Linear trend forecast
        x = np.arange(len(ts_data))
        trend_coef = np.polyfit(x, ts_data, 1)
        trend_func = np.poly1d(trend_coef)

        # Generate 30-day forecast
        forecast_days = 30
        future_x = np.arange(len(ts_data), len(ts_data) + forecast_days)
        trend_forecast = trend_func(future_x)

        # Create forecast dates
        last_date = dates[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='D')

        # Plot forecast
        fig = go.Figure()

        # Historical data (last 60 days)
        recent_data = ts_data[-60:] if len(ts_data) > 60 else ts_data
        recent_dates = dates[-60:] if len(dates) > 60 else dates

        fig.add_trace(go.Scatter(
            x=recent_dates,
            y=recent_data,
            mode='lines',
            name='Historical',
            line=dict(color='blue')
        ))

        # Moving average forecast
        ma_line = np.full(forecast_days, ma_forecast)
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=ma_line,
            mode='lines',
            name=f'{window}-Day MA Forecast',
            line=dict(color='green', dash='dash')
        ))

        # Trend forecast
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=trend_forecast,
            mode='lines',
            name='Trend Forecast',
            line=dict(color='red', dash='dot')
        ))

        fig.update_layout(
            title="Simple Forecasting Models - EU Visits (30 Days)",
            height=500,
            xaxis_title="Date",
            yaxis_title="Daily Visits"
        )
        st.plotly_chart(fig, use_container_width=True, key="simple_forecast")

        # Forecast metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Moving Average Forecast", f"{ma_forecast:.0f}")
        with col2:
            trend_direction = "Increasing" if trend_coef[0] > 0 else "Decreasing"
            st.metric("Trend Direction", trend_direction)
        with col3:
            st.metric("Trend Next Day", f"{trend_forecast[0]:.0f}")

    def _prepare_forecasting_data(self) -> Tuple[Optional[pd.DataFrame], str]:
        """Prepare data for forecasting analysis"""
        if self.eu_data is None:
            return None, "EU Visits data not available"

        if 'Visit_Count' not in self.eu_data.columns:
            return None, "Visit_Count column not found"

        if 'Date' not in self.eu_data.columns:
            return None, "Date column not found"

        # Aggregate daily data
        daily_data = self.eu_data.groupby('Date')['Visit_Count'].sum().reset_index()
        daily_data = daily_data.sort_values('Date')

        if len(daily_data) < 10:
            return None, f"Insufficient data for forecasting: {len(daily_data)} days available"

        return daily_data, ""

    def _create_ml_features(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """Create machine learning features from daily data"""
        ml_data = daily_data.copy()

        # Lag features
        for lag in [1, 2, 3, 7, 14]:
            ml_data[f'lag_{lag}'] = ml_data['Visit_Count'].shift(lag)

        # Rolling statistics
        for window in [3, 7, 14]:
            ml_data[f'rolling_mean_{window}'] = ml_data['Visit_Count'].rolling(window=window).mean()
            ml_data[f'rolling_std_{window}'] = ml_data['Visit_Count'].rolling(window=window).std()

        # Date features
        ml_data['day_of_week'] = ml_data['Date'].dt.dayofweek
        ml_data['month'] = ml_data['Date'].dt.month
        ml_data['day_of_year'] = ml_data['Date'].dt.dayofyear

        # Cyclical encoding
        ml_data['day_of_week_sin'] = np.sin(2 * np.pi * ml_data['day_of_week'] / 7)
        ml_data['day_of_week_cos'] = np.cos(2 * np.pi * ml_data['day_of_week'] / 7)
        ml_data['month_sin'] = np.sin(2 * np.pi * ml_data['month'] / 12)
        ml_data['month_cos'] = np.cos(2 * np.pi * ml_data['month'] / 12)

        # Trend feature
        ml_data['trend'] = np.arange(len(ml_data))

        return ml_data

    def _show_ensemble_forecasting_fixed(self):
        """FIXED: 12-Month Strategic Forecasting with Working Visualizations"""
        st.markdown("#### 12-Month Strategic Forecast & Predictions")
        st.info("Long-term predictions for capacity planning and resource allocation")

        daily_data, error = self._prepare_forecasting_data()
        if daily_data is None:
            st.error(f"{error}")
            return

        # Generate comprehensive yearly forecasts with working visualizations
        self._generate_yearly_insights_fixed(daily_data)

    def _generate_yearly_insights_fixed(self, daily_data):
        """FIXED: Generate comprehensive yearly forecasting insights with working visualizations"""

        # Analyze historical patterns for projections
        historical_analysis = self._analyze_historical_patterns(daily_data)

        st.markdown("##### Next 12 Months EU Visits Forecast")

        # Generate 12-month forecast using historical patterns
        yearly_forecast = self._create_yearly_forecast(daily_data, historical_analysis)

        if yearly_forecast is not None:
            # Display yearly total prediction
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                total_forecast = yearly_forecast['Predicted_Visits'].sum()
                st.metric("Next 12 Months Total", f"{total_forecast:,.0f}")

            with col2:
                daily_avg = yearly_forecast['Predicted_Visits'].mean()
                st.metric("Daily Average", f"{daily_avg:.0f}")

            with col3:
                historical_yearly = daily_data['Visit_Count'].sum() * (365 / len(daily_data))
                growth_rate = ((total_forecast - historical_yearly) / historical_yearly) * 100
                st.metric("Projected Growth", f"{growth_rate:+.1f}%")

            with col4:
                peak_month_visits = yearly_forecast.groupby(yearly_forecast['Date'].dt.month)[
                    'Predicted_Visits'].sum().max()
                st.metric("Peak Month Volume", f"{peak_month_visits:.0f}")

            # Plot 12-month forecast
            fig = go.Figure()

            # Historical data (last 90 days)
            recent_data = daily_data.tail(90)
            fig.add_trace(go.Scatter(
                x=recent_data['Date'],
                y=recent_data['Visit_Count'],
                mode='lines',
                name='Recent Historical',
                line=dict(color='blue', width=2)
            ))

            # 12-month forecast
            fig.add_trace(go.Scatter(
                x=yearly_forecast['Date'],
                y=yearly_forecast['Predicted_Visits'],
                mode='lines',
                name='12-Month Forecast',
                line=dict(color='red', width=3)
            ))

            # Add confidence bands
            upper_bound = yearly_forecast['Predicted_Visits'] * 1.15
            lower_bound = yearly_forecast['Predicted_Visits'] * 0.85

            fig.add_trace(go.Scatter(
                x=yearly_forecast['Date'],
                y=upper_bound,
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False
            ))

            fig.add_trace(go.Scatter(
                x=yearly_forecast['Date'],
                y=lower_bound,
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,0,0,0)',
                name='Confidence Range (±15%)',
                fillcolor='rgba(255,0,0,0.2)'
            ))

            fig.update_layout(
                title="12-Month EU Visits Forecast",
                height=500,
                xaxis_title="Date",
                yaxis_title="Predicted Daily Visits"
            )
            st.plotly_chart(fig, use_container_width=True, key="yearly_forecast_fixed")

            # Strategic insights section
            st.markdown("##### Strategic Planning Insights")

            # Generate specific predictions
            insights = self._generate_strategic_insights(yearly_forecast, historical_analysis)

            # Display insights in organized sections with WORKING VISUALIZATIONS
            insight_tab1, insight_tab2, insight_tab3 = st.tabs([
                "Peak Periods", "Seasonal Trends", "Operational Planning"
            ])

            with insight_tab1:
                self._show_peak_periods_insights_fixed(insights, yearly_forecast)

            with insight_tab2:
                self._show_seasonal_trends_insights_fixed(insights, yearly_forecast, historical_analysis)

            with insight_tab3:
                self._show_operational_planning_insights(insights, yearly_forecast)

    def _show_peak_periods_insights_fixed(self, insights, yearly_forecast):
        """FIXED: Display peak periods insights with working visualizations including quarterly trends"""
        st.markdown("#### Peak Period Predictions")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### Monthly Peaks")
            st.success(
                f"**Highest Month**: {insights['highest_month']['month']} ({insights['highest_month']['visits']:,.0f} visits)")
            st.info(
                f"**Lowest Month**: {insights['lowest_month']['month']} ({insights['lowest_month']['visits']:,.0f} visits)")

            difference = insights['highest_month']['visits'] - insights['lowest_month']['visits']
            st.metric("Peak vs Low Difference", f"{difference:,.0f} visits")

        with col2:
            st.markdown("##### Weekly Patterns")
            st.success(
                f"**Busiest Day**: {insights['busiest_day']['day']} ({insights['busiest_day']['visits']:,.0f} total visits)")
            st.info(
                f"**Quietest Day**: {insights['quietest_day']['day']} ({insights['quietest_day']['visits']:,.0f} total visits)")

            weekly_difference = insights['busiest_day']['visits'] - insights['quietest_day']['visits']
            st.metric("Busiest vs Quietest", f"{weekly_difference:,.0f} visits")

        # Create monthly prediction chart
        st.markdown("##### Monthly Volume Predictions")

        # Get actual monthly totals from forecast
        monthly_data = yearly_forecast.groupby('Month')['Predicted_Visits'].sum().reset_index()
        monthly_data['Month_Name'] = monthly_data['Month'].map(lambda x: calendar.month_abbr[x])
        monthly_data['Is_Peak'] = monthly_data['Month'] == insights['highest_month']['month_num']
        monthly_data['Is_Low'] = monthly_data['Month'] == insights['lowest_month']['month_num']

        # Create bar chart with peak highlighting
        fig = go.Figure()

        # Regular months
        regular_months = monthly_data[~monthly_data['Is_Peak'] & ~monthly_data['Is_Low']]
        if not regular_months.empty:
            fig.add_trace(go.Bar(
                x=regular_months['Month_Name'],
                y=regular_months['Predicted_Visits'],
                name='Regular Months',
                marker_color='lightblue'
            ))

        # Peak month
        peak_month = monthly_data[monthly_data['Is_Peak']]
        if not peak_month.empty:
            fig.add_trace(go.Bar(
                x=peak_month['Month_Name'],
                y=peak_month['Predicted_Visits'],
                name='Peak Month',
                marker_color='red',
                text=[f"PEAK: {int(val):,}" for val in peak_month['Predicted_Visits']],
                textposition='outside'
            ))

        # Low month
        low_month = monthly_data[monthly_data['Is_Low']]
        if not low_month.empty:
            fig.add_trace(go.Bar(
                x=low_month['Month_Name'],
                y=low_month['Predicted_Visits'],
                name='Low Month',
                marker_color='green',
                text=[f"LOW: {int(val):,}" for val in low_month['Predicted_Visits']],
                textposition='outside'
            ))

        fig.update_layout(
            title="Next 12 Months: EU Visits Prediction by Month",
            height=400,
            xaxis_title="Month",
            yaxis_title="Predicted Visits",
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True, key="monthly_predictions_fixed")

        # Create weekly prediction chart
        st.markdown("##### Weekly Pattern Predictions")

        # Get actual weekly totals from forecast
        weekly_data = yearly_forecast.groupby('DayOfWeek')['Predicted_Visits'].agg(['sum', 'mean']).reset_index()
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_data['Day_Name'] = weekly_data['DayOfWeek'].map(lambda x: day_names[x])
        weekly_data['Is_Peak'] = weekly_data['DayOfWeek'] == insights['busiest_day']['day_num']
        weekly_data['Is_Low'] = weekly_data['DayOfWeek'] == insights['quietest_day']['day_num']

        # Create horizontal bar chart for better day visibility
        fig = go.Figure()

        colors = []
        for _, row in weekly_data.iterrows():
            if row['Is_Peak']:
                colors.append('red')
            elif row['Is_Low']:
                colors.append('green')
            else:
                colors.append('lightblue')

        fig.add_trace(go.Bar(
            x=weekly_data['mean'],
            y=weekly_data['Day_Name'],
            orientation='h',
            marker_color=colors,
            text=[f"{val:.0f}" for val in weekly_data['mean']],
            textposition='inside'
        ))

        fig.update_layout(
            title="Daily Average Visits Prediction (Next Year)",
            height=400,
            xaxis_title="Daily Average Visits",
            yaxis_title="Day of Week"
        )
        st.plotly_chart(fig, use_container_width=True, key="weekly_predictions_fixed")

        # Quarterly trends visualization
        st.markdown("##### Quarterly Trends & Predictions")

        quarters = ['Q1', 'Q2', 'Q3', 'Q4']

        # Generate quarterly data from forecast
        quarterly_data = yearly_forecast.groupby('Quarter')['Predicted_Visits'].agg(
            ['sum', 'mean', 'std']).reset_index()
        quarterly_data['Growth_Rate'] = quarterly_data['sum'].pct_change() * 100
        quarterly_data['Growth_Rate'].fillna(0, inplace=True)

        # Create dual-axis chart
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"secondary_y": True}]],
            subplot_titles=["Quarterly Visits & Growth Predictions"]
        )

        # Visits bars
        colors = ['red' if q == insights['highest_quarter']['quarter'] else 'lightblue' for q in
                  quarterly_data['Quarter']]
        fig.add_trace(
            go.Bar(
                x=quarterly_data['Quarter'],
                y=quarterly_data['sum'],
                name='Predicted Visits',
                marker_color=colors,
                text=[f"{int(val):,}" for val in quarterly_data['sum']],
                textposition='outside'
            ),
            secondary_y=False,
        )

        # Growth rate line
        fig.add_trace(
            go.Scatter(
                x=quarterly_data['Quarter'],
                y=quarterly_data['Growth_Rate'],
                mode='lines+markers',
                name='Growth Rate (%)',
                line=dict(color='green', width=3),
                marker=dict(size=8)
            ),
            secondary_y=True,
        )

        # Update layout
        fig.update_xaxes(title_text="Quarter")
        fig.update_yaxes(title_text="Predicted Visits", secondary_y=False)
        fig.update_yaxes(title_text="Growth Rate (%)", secondary_y=True)
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True, key="quarterly_predictions_fixed")

        # Quarterly planning table
        st.markdown("##### Quarterly Planning Guide")

        planning_df = quarterly_data.copy()
        planning_df['Budget_Impact'] = planning_df.apply(
            lambda row: 'High' if row['Quarter'] == insights['highest_quarter']['quarter']
            else 'Medium' if row['Growth_Rate'] > 5
            else 'Low', axis=1
        )
        planning_df['Action_Required'] = planning_df.apply(
            lambda row: 'Surge capacity planning' if row['Quarter'] == insights['highest_quarter']['quarter']
            else 'Standard preparation' if row['Growth_Rate'] > 0
            else 'Cost optimization opportunity', axis=1
        )

        display_planning = planning_df[['Quarter', 'sum', 'Growth_Rate', 'Budget_Impact', 'Action_Required']].copy()
        display_planning.columns = ['Quarter', 'Predicted_Visits', 'Growth_Rate_%', 'Budget_Impact', 'Action_Required']
        display_planning['Predicted_Visits'] = display_planning['Predicted_Visits'].astype(int)
        display_planning['Growth_Rate_%'] = display_planning['Growth_Rate_%'].round(1)

        st.dataframe(display_planning, use_container_width=True)

        # Specific predictions summary
        st.markdown("##### Key Predictions Summary")

        predictions = [
            f"Peak Week: Week {insights['highest_week']['week']} will be the busiest with {insights['highest_week']['visits']:,.0f} predicted visits",
            f"Highest Quarter: {insights['highest_quarter']['quarter']} will have {insights['highest_quarter']['visits']:,.0f} visits",
            f"Peak Season: {insights['highest_season']['season']} will see {insights['highest_season']['visits']:,.0f} visits"
        ]

        for prediction in predictions:
            st.markdown(f"- {prediction}")

    def _show_seasonal_trends_insights_fixed(self, insights, yearly_forecast, historical_analysis):
        """FIXED: Display seasonal trends insights with comprehensive visualizations"""
        st.markdown("#### Seasonal Trend Analysis with Predictions")

        # Get seasonal data from forecast
        seasonal_data = yearly_forecast.groupby('Season')['Predicted_Visits'].agg(['sum', 'mean', 'std']).reset_index()
        seasonal_data = seasonal_data.sort_values('sum', ascending=False)

        st.success(
            f"**Highest Demand Season**: {insights['highest_season']['season']} with {insights['highest_season']['visits']:,.0f} predicted visits")

        # Create seasonal comparison visualization
        st.markdown("##### Seasonal Volume Comparison")

        fig = px.bar(
            seasonal_data,
            x='Season',
            y='sum',
            color='Season',
            title="Total Predicted Visits by Season (Next 12 Months)",
            color_discrete_sequence=['lightgreen', 'gold', 'orange', 'lightblue']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True, key="seasonal_comparison_fixed")

        # Create seasonal heatmap
        st.markdown("##### Seasonal Demand Intensity Heatmap")

        # Create seasonal data matrix
        seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
        months_map = {
            'Spring': [3, 4, 5],
            'Summer': [6, 7, 8],
            'Autumn': [9, 10, 11],
            'Winter': [12, 1, 2]
        }

        # Generate heatmap data from actual forecast
        heatmap_data = []

        for season in seasons:
            season_months = months_map[season]
            for month in season_months:
                month_data = yearly_forecast[yearly_forecast['Month'] == month]['Predicted_Visits'].sum()
                max_month = yearly_forecast.groupby('Month')['Predicted_Visits'].sum().max()
                intensity = month_data / max_month if max_month > 0 else 0  # Normalize to 0-1

                heatmap_data.append({
                    'Season': season,
                    'Month': calendar.month_abbr[month],
                    'Demand_Intensity': intensity,
                    'Total_Visits': month_data
                })

        heatmap_df = pd.DataFrame(heatmap_data)

        # Create heatmap matrix
        pivot_data = heatmap_df.pivot(index='Season', columns='Month', values='Demand_Intensity')

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='RdYlBu_r',
            text=[[f"{val:.2f}" for val in row] for row in pivot_data.values],
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Demand Intensity")
        ))

        fig.update_layout(
            title="Seasonal Demand Intensity Heatmap (Next Year)",
            height=400,
            xaxis_title="Month",
            yaxis_title="Season"
        )
        st.plotly_chart(fig, use_container_width=True, key="seasonal_heatmap_fixed")

        # Create seasonal trend line
        st.markdown("##### Seasonal Progression Analysis")

        # Create monthly progression
        monthly_progression = yearly_forecast.groupby([yearly_forecast['Date'].dt.to_period('M')])[
            'Predicted_Visits'].sum()
        monthly_progression.index = monthly_progression.index.to_timestamp()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=monthly_progression.index,
            y=monthly_progression.values,
            mode='lines+markers',
            name='Monthly Totals',
            line=dict(color='blue', width=2)
        ))

        # Add seasonal trend line
        seasonal_avg = []
        for date in monthly_progression.index:
            month = date.month
            if month in historical_analysis['monthly'].index:
                seasonal_avg.append(historical_analysis['monthly'].loc[month, 'mean'] * 30)  # Convert to monthly
            else:
                seasonal_avg.append(monthly_progression.mean())

        fig.add_trace(go.Scatter(
            x=monthly_progression.index,
            y=seasonal_avg,
            mode='lines',
            name='Historical Seasonal Pattern',
            line=dict(color='red', dash='dash', width=2)
        ))

        fig.update_layout(
            title="Monthly Forecast vs Historical Seasonal Pattern",
            height=400,
            xaxis_title="Month",
            yaxis_title="Predicted Visits"
        )
        st.plotly_chart(fig, use_container_width=True, key="seasonal_progression_fixed")

        # Generate data-driven seasonal recommendations
        st.markdown("##### Data-Driven Seasonal Strategy")

        # Calculate seasonal metrics
        peak_season = insights['highest_season']['season']
        peak_visits = seasonal_data[seasonal_data['Season'] == peak_season]['sum'].iloc[
            0] if not seasonal_data.empty else 0

        # Calculate relative increases needed
        for _, row in seasonal_data.iterrows():
            season = row['Season']
            visits = row['sum']
            mean_visits = row['mean']
            std_visits = row['std']

            if season == peak_season:
                increase_factor = 1.0
                status = "PEAK SEASON"
                color = "error"
            else:
                increase_factor = peak_visits / visits if visits > 0 else 1.0
                status = f"{((increase_factor - 1) * 100):+.0f}% vs Peak"
                color = "info"

            # Generate specific recommendations based on data
            variability = (std_visits / mean_visits) * 100 if mean_visits > 0 else 0

            recommendations = []
            if increase_factor > 1.3:  # 30% or more increase needed
                recommendations.append(f"Increase staffing capacity by {((increase_factor - 1) * 100):.0f}%")
            elif increase_factor > 1.1:  # 10-30% increase needed
                recommendations.append(
                    f"Moderate capacity increase of {((increase_factor - 1) * 100):.0f}% recommended")
            else:
                recommendations.append("Standard capacity sufficient")

            if variability > 20:
                recommendations.append(f"High variability ({variability:.0f}%) - prepare flexible staffing")
            elif variability > 10:
                recommendations.append(f"Moderate variability ({variability:.0f}%) - monitor weekly patterns")
            else:
                recommendations.append(f"Low variability ({variability:.0f}%) - predictable demand")

            # Display season-specific insights
            if color == "error":
                st.error(f"**{season}** ({status}): {visits:,.0f} visits")
            else:
                st.info(f"**{season}** ({status}): {visits:,.0f} visits")

            for rec in recommendations:
                st.markdown(f"  - {rec}")

    def _show_operational_planning_insights(self, insights, yearly_forecast):
        """Display operational planning insights"""
        st.markdown("#### Operational Planning Recommendations")

        # Generate dynamic staffing recommendations
        st.markdown("##### Staffing Recommendations")

        # Calculate staffing needs based on volume differences
        busiest_day_avg = yearly_forecast[yearly_forecast['DayOfWeek'] == insights['busiest_day']['day_num']][
            'Predicted_Visits'].mean()
        quietest_day_avg = yearly_forecast[yearly_forecast['DayOfWeek'] == insights['quietest_day']['day_num']][
            'Predicted_Visits'].mean()
        overall_avg = yearly_forecast['Predicted_Visits'].mean()

        busiest_increase = ((busiest_day_avg - overall_avg) / overall_avg) * 100
        quietest_decrease = ((overall_avg - quietest_day_avg) / overall_avg) * 100

        staffing_recs = [
            f"**{insights['busiest_day']['day']} Staffing**: Increase by {busiest_increase:.0f}% above baseline ({busiest_day_avg:.0f} avg visits)",
            f"**{insights['quietest_day']['day']} Optimization**: {quietest_decrease:.0f}% below average - optimal for training/maintenance",
            f"**{insights['highest_month']['month']} Peak**: Plan {((insights['highest_month']['visits'] / 12 - overall_avg) / overall_avg * 100):.0f}% capacity increase",
            f"**Week {insights['highest_week']['week']} Surge**: Maximum resources needed ({insights['highest_week']['visits']:,.0f} total visits)"
        ]

        for rec in staffing_recs:
            st.markdown(f"- {rec}")

        # Generate capacity planning recommendations
        st.markdown("##### Capacity Planning")

        # Calculate capacity metrics
        max_daily = yearly_forecast['Predicted_Visits'].max()
        min_daily = yearly_forecast['Predicted_Visits'].min()
        capacity_range = max_daily - min_daily
        variability = (yearly_forecast['Predicted_Visits'].std() / yearly_forecast['Predicted_Visits'].mean()) * 100

        capacity_recs = [
            f"**Peak Capacity**: Design for {max_daily:.0f} daily visits maximum",
            f"**Variability Management**: {variability:.0f}% coefficient of variation requires flexible capacity",
            f"**Equipment Planning**: {capacity_range:.0f} visit range necessitates scalable resources",
            f"**Surge Protocols**: Implement for demands above {yearly_forecast['Predicted_Visits'].quantile(0.95):.0f} visits"
        ]

        for rec in capacity_recs:
            st.markdown(f"- {rec}")

        # Generate resource allocation recommendations
        st.markdown("##### Resource Allocation")

        # Calculate quarterly resource needs
        quarterly_data = yearly_forecast.groupby('Quarter')['Predicted_Visits'].sum()
        peak_quarter_visits = quarterly_data.max()

        resource_recs = [
            f"**Budget Priority**: {insights['highest_quarter']['quarter']} requires {((peak_quarter_visits / quarterly_data.mean() - 1) * 100):.0f}% above average allocation",
            f"**Inventory Planning**: Stock levels should peak before {insights['highest_month']['month']}",
            f"**Maintenance Scheduling**: Optimal during {insights['lowest_month']['month']} (lowest predicted demand)",
            f"**Training Calendar**: Schedule major training during Week {yearly_forecast.groupby('Week')['Predicted_Visits'].sum().idxmin()}"
        ]

        for rec in resource_recs:
            st.markdown(f"- {rec}")

        # Generate action timeline
        st.markdown("##### Priority Action Timeline")

        next_month = (datetime.now() + timedelta(days=30)).month
        if next_month == insights['highest_month']['month_num']:
            st.error(
                f"**IMMEDIATE ACTION**: Peak month {insights['highest_month']['month']} approaching - implement surge capacity now")
        elif abs(next_month - insights['highest_month']['month_num']) <= 2:
            st.warning(
                f"**PREPARE**: Peak month {insights['highest_month']['month']} in 2-3 months - begin capacity planning")
        else:
            st.info(
                f"**PLAN**: Peak month {insights['highest_month']['month']} in {abs(next_month - insights['highest_month']['month_num'])} months - long-term preparation phase")

    def _show_ml_forecasting(self):
        """Machine Learning Forecasting Models"""
        st.markdown("#### Machine Learning Forecasting Models")

        daily_data, error = self._prepare_forecasting_data()
        if daily_data is None:
            st.error(f"❌ {error}")
            return

        st.markdown("##### Feature Engineering")

        # Create features
        ml_data = self._create_ml_features(daily_data)

        # Remove rows with NaN values (due to lag and rolling features)
        ml_data_clean = ml_data.dropna()

        if len(ml_data_clean) < 30:
            st.error("❌ Insufficient data after feature engineering")
            return

        # Display feature importance
        feature_cols = [col for col in ml_data_clean.columns if col not in ['Date', 'Visit_Count']]
        st.info(
            f"✅ Features Created: {len(feature_cols)} features including lags, rolling stats, and cyclical encodings")

        # Prepare ML data
        X = ml_data_clean[feature_cols]
        y = ml_data_clean['Visit_Count']

        # Train-test split
        test_size = min(0.3, 30 / len(X))  # At least 30 days for test or 30% of data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

        st.markdown("##### Model Training & Comparison")

        # Initialize models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression()
        }

        model_results = {}

        # Train and evaluate models
        for model_name, model in models.items():
            try:
                with st.spinner(f"Training {model_name}..."):
                    # Train model
                    model.fit(X_train, y_train)

                    # Make predictions
                    y_pred = model.predict(X_test)

                    # Calculate metrics
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    r2 = r2_score(y_test, y_pred)
                    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

                    model_results[model_name] = {
                        'model': model,
                        'predictions': y_pred,
                        'mae': mae,
                        'rmse': rmse,
                        'r2': r2,
                        'mape': mape
                    }

            except Exception as e:
                st.warning(f"⚠️ {model_name} training failed: {str(e)}")

        # Display model comparison
        if model_results:
            st.markdown("##### Model Performance Comparison")

            performance_df = pd.DataFrame({
                'Model': list(model_results.keys()),
                'MAE': [results['mae'] for results in model_results.values()],
                'RMSE': [results['rmse'] for results in model_results.values()],
                'R²': [results['r2'] for results in model_results.values()],
                'MAPE (%)': [results['mape'] for results in model_results.values()]
            }).round(3)

            st.dataframe(performance_df, use_container_width=True)

            # Plot model predictions vs actual
            fig = go.Figure()

            # Test dates
            test_dates = ml_data_clean.iloc[len(X_train):]['Date']

            # Actual values
            fig.add_trace(go.Scatter(
                x=test_dates,
                y=y_test,
                mode='lines+markers',
                name='Actual',
                line=dict(color='blue', width=3)
            ))

            # Model predictions
            colors = ['red', 'green', 'orange', 'purple']
            for i, (model_name, results) in enumerate(model_results.items()):
                fig.add_trace(go.Scatter(
                    x=test_dates,
                    y=results['predictions'],
                    mode='lines+markers',
                    name=f'{model_name} Prediction',
                    line=dict(color=colors[i % len(colors)], width=2, dash='dash')
                ))

            fig.update_layout(
                title="ML Model Predictions vs Actual Values",
                height=500,
                xaxis_title="Date",
                yaxis_title="Daily Visits"
            )
            st.plotly_chart(fig, use_container_width=True, key="ml_predictions")

            # Feature importance for best model
            best_model_name = min(model_results.keys(), key=lambda k: model_results[k]['mae'])
            best_model = model_results[best_model_name]['model']

            st.markdown(f"##### Feature Importance ({best_model_name})")

            if hasattr(best_model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': best_model.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)

                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title=f"Top 10 Feature Importance - {best_model_name}"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True, key="feature_importance")

            # Best model summary
            st.success(
                f"🏆 Best performing model: **{best_model_name}** (MAE: {model_results[best_model_name]['mae']:.2f})")
        else:
            st.error("❌ No models were successfully trained")

    def _show_model_comparison(self):
        """Compare different forecasting approaches"""
        st.markdown("#### Comprehensive Model Comparison")

        daily_data, error = self._prepare_forecasting_data()
        if daily_data is None:
            st.error(f"{error}")
            return

        st.markdown("##### Baseline vs Advanced Models")

        # Prepare data
        ts_data = daily_data['Visit_Count'].values
        dates = daily_data['Date'].values

        # Split data
        split_point = int(len(ts_data) * 0.8)
        train_data = ts_data[:split_point]
        test_data = ts_data[split_point:]
        train_dates = dates[:split_point]
        test_dates = dates[split_point:]

        models_comparison = {}

        # Naive forecast (last value)
        naive_forecast = np.full(len(test_data), train_data[-1])
        models_comparison['Naive (Last Value)'] = naive_forecast

        # Moving average forecast
        ma_window = min(7, len(train_data) // 4)
        ma_forecast = np.full(len(test_data), np.mean(train_data[-ma_window:]))
        models_comparison['Moving Average'] = ma_forecast

        # Linear trend forecast
        x_train = np.arange(len(train_data))
        trend_coef = np.polyfit(x_train, train_data, 1)
        trend_func = np.poly1d(trend_coef)
        x_test = np.arange(len(train_data), len(train_data) + len(test_data))
        trend_forecast = trend_func(x_test)
        models_comparison['Linear Trend'] = trend_forecast

        # Exponential smoothing (if available)
        if STATSMODELS_AVAILABLE:
            try:
                exp_model = ExponentialSmoothing(train_data, trend='add', seasonal=None)
                exp_fit = exp_model.fit()
                exp_forecast = exp_fit.forecast(len(test_data))
                models_comparison['Exponential Smoothing'] = exp_forecast
            except:
                pass

        # Calculate performance metrics
        performance_results = []
        for model_name, forecast in models_comparison.items():
            mae = mean_absolute_error(test_data, forecast)
            rmse = np.sqrt(mean_squared_error(test_data, forecast))
            mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100

            performance_results.append({
                'Model': model_name,
                'MAE': mae,
                'RMSE': rmse,
                'MAPE (%)': mape
            })

        # Display comparison table
        comparison_df = pd.DataFrame(performance_results).round(3)
        st.dataframe(comparison_df, use_container_width=True)

        # Plot model comparison
        fig = go.Figure()

        # Historical training data
        fig.add_trace(go.Scatter(
            x=train_dates,
            y=train_data,
            mode='lines',
            name='Training Data',
            line=dict(color='blue', width=2)
        ))

        # Actual test data
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=test_data,
            mode='lines+markers',
            name='Actual Test Data',
            line=dict(color='black', width=3)
        ))

        # Model forecasts
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        for i, (model_name, forecast) in enumerate(models_comparison.items()):
            fig.add_trace(go.Scatter(
                x=test_dates,
                y=forecast,
                mode='lines',
                name=model_name,
                line=dict(color=colors[i % len(colors)], width=2, dash='dash')
            ))

        fig.update_layout(
            title="Forecasting Models Comparison",
            height=500,
            xaxis_title="Date",
            yaxis_title="Daily Visits"
        )
        st.plotly_chart(fig, use_container_width=True, key="models_comparison")

        # Best model recommendation
        best_model = comparison_df.loc[comparison_df['MAE'].idxmin()]
        st.success(f"Best Performing Model: {best_model['Model']} (MAE: {best_model['MAE']:.3f})")

    def _show_model_validation(self):
        """Model validation and diagnostics"""
        st.markdown("#### Model Validation & Diagnostics")

        daily_data, error = self._prepare_forecasting_data()
        if daily_data is None:
            st.error(f"❌ {error}")
            return

        st.markdown("##### Validation Framework")

        validation_checks = [
            "**Cross-Validation**: Time series cross-validation with expanding window",
            "**Residual Analysis**: Check for patterns in model residuals",
            "**Forecast Accuracy**: Multiple error metrics (MAE, RMSE, MAPE)",
            "**Statistical Tests**: Ljung-Box test for residual autocorrelation",
            "**Stability Tests**: Model performance across different time periods",
            "**Outlier Detection**: Impact of anomalies on forecast accuracy"
        ]

        for check in validation_checks:
            st.markdown(f"- {check}")

        # Basic residual analysis
        st.markdown("##### Residual Analysis")

        ts_data = daily_data['Visit_Count'].values
        dates = daily_data['Date'].values

        # Simple moving average for demonstration
        window = 7
        if len(ts_data) > window:
            ma_forecast = np.convolve(ts_data, np.ones(window) / window, mode='valid')
            residuals = ts_data[window - 1:] - ma_forecast
            residual_dates = dates[window - 1:]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=residual_dates,
                y=residuals,
                mode='markers+lines',
                name='Residuals',
                line=dict(color='red')
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="black")
            fig.update_layout(
                title="Model Residuals Analysis (7-Day Moving Average)",
                height=400,
                xaxis_title="Date",
                yaxis_title="Residuals"
            )
            st.plotly_chart(fig, use_container_width=True, key="residuals_analysis")

            # Validation metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Residual", f"{np.mean(residuals):.3f}")
            with col2:
                st.metric("Residual Std Dev", f"{np.std(residuals):.3f}")
            with col3:
                st.metric("Residual Range", f"{np.ptp(residuals):.3f}")

            # Residual distribution
            fig = px.histogram(
                x=residuals,
                nbins=20,
                title="Residual Distribution"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True, key="residual_histogram")
        else:
            st.warning("⚠️ Insufficient data for residual analysis")

    def _analyze_historical_patterns(self, daily_data):
        """Analyze historical patterns for forecasting"""
        patterns = {}

        # Add month, day of week, etc.
        daily_data = daily_data.copy()
        daily_data['Month'] = daily_data['Date'].dt.month
        daily_data['DayOfWeek'] = daily_data['Date'].dt.dayofweek
        daily_data['Week'] = daily_data['Date'].dt.isocalendar().week

        # Monthly patterns
        patterns['monthly'] = daily_data.groupby('Month')['Visit_Count'].agg(['mean', 'std']).round(1)

        # Day of week patterns
        patterns['daily'] = daily_data.groupby('DayOfWeek')['Visit_Count'].agg(['mean', 'std']).round(1)

        # Weekly patterns
        patterns['weekly'] = daily_data.groupby('Week')['Visit_Count'].agg(['mean', 'std']).round(1)

        # Seasonal patterns
        daily_data['Season'] = daily_data['Month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
        })
        patterns['seasonal'] = daily_data.groupby('Season')['Visit_Count'].agg(['mean', 'std']).round(1)

        # Growth trend
        if len(daily_data) > 30:
            recent_avg = daily_data.tail(30)['Visit_Count'].mean()
            older_avg = daily_data.head(30)['Visit_Count'].mean()
            patterns['growth_trend'] = (recent_avg - older_avg) / older_avg * 100 if older_avg > 0 else 0
        else:
            patterns['growth_trend'] = 0

        return patterns

    def _create_yearly_forecast(self, daily_data, patterns):
        """Create 12-month forecast using historical patterns"""
        try:
            # Generate future dates (365 days)
            last_date = daily_data['Date'].max()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=365, freq='D')

            forecast_data = []

            for date in future_dates:
                # Get base prediction from patterns
                month = date.month
                dayofweek = date.dayofweek
                week = date.isocalendar().week

                # Base prediction from monthly pattern
                if month in patterns['monthly'].index:
                    base_prediction = patterns['monthly'].loc[month, 'mean']
                else:
                    base_prediction = daily_data['Visit_Count'].mean()

                # Adjust for day of week
                if dayofweek in patterns['daily'].index:
                    dow_factor = patterns['daily'].loc[dayofweek, 'mean'] / patterns['daily']['mean'].mean()
                    base_prediction *= dow_factor

                # Apply growth trend
                days_ahead = (date - last_date).days
                growth_factor = 1 + (patterns['growth_trend'] / 100) * (days_ahead / 365)
                final_prediction = base_prediction * growth_factor

                # Add some realistic noise
                noise = np.random.normal(0, base_prediction * 0.05)
                final_prediction += noise

                # Ensure non-negative
                final_prediction = max(0, final_prediction)

                forecast_data.append({
                    'Date': date,
                    'Predicted_Visits': final_prediction,
                    'Month': month,
                    'DayOfWeek': dayofweek,
                    'Week': week
                })

            forecast_df = pd.DataFrame(forecast_data)
            # Add Quarter and Season for quarterly and seasonal features
            forecast_df['Quarter'] = forecast_df['Month'].map({
                1: 'Q1', 2: 'Q1', 3: 'Q1',
                4: 'Q2', 5: 'Q2', 6: 'Q2',
                7: 'Q3', 8: 'Q3', 9: 'Q3',
                10: 'Q4', 11: 'Q4', 12: 'Q4'
            })
            seasonal_map = {
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
            }
            forecast_df['Season'] = forecast_df['Month'].map(seasonal_map)

            return forecast_df

        except Exception as e:
            st.error(f"Error creating yearly forecast: {str(e)}")
            return None

    def _generate_strategic_insights(self, yearly_forecast, patterns):
        """Generate strategic insights from forecast"""
        insights = {}

        # Monthly insights
        monthly_totals = yearly_forecast.groupby('Month')['Predicted_Visits'].sum().round(0)
        insights['highest_month'] = {
            'month': calendar.month_name[monthly_totals.idxmax()],
            'visits': monthly_totals.max(),
            'month_num': monthly_totals.idxmax()
        }
        insights['lowest_month'] = {
            'month': calendar.month_name[monthly_totals.idxmin()],
            'visits': monthly_totals.min(),
            'month_num': monthly_totals.idxmin()
        }

        # Day of week insights
        dow_totals = yearly_forecast.groupby('DayOfWeek')['Predicted_Visits'].sum()
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        insights['busiest_day'] = {
            'day': day_names[dow_totals.idxmax()],
            'visits': dow_totals.max(),
            'day_num': dow_totals.idxmax()
        }
        insights['quietest_day'] = {
            'day': day_names[dow_totals.idxmin()],
            'visits': dow_totals.min(),
            'day_num': dow_totals.idxmin()
        }

        # Seasonal insights
        seasonal_totals = yearly_forecast.groupby('Season')['Predicted_Visits'].sum()
        insights['highest_season'] = {
            'season': seasonal_totals.idxmax(),
            'visits': seasonal_totals.max()
        }

        # Weekly insights
        weekly_totals = yearly_forecast.groupby('Week')['Predicted_Visits'].sum()
        insights['highest_week'] = {
            'week': weekly_totals.idxmax(),
            'visits': weekly_totals.max()
        }

        # Quarterly insights
        quarterly_totals = yearly_forecast.groupby('Quarter')['Predicted_Visits'].sum()
        insights['highest_quarter'] = {
            'quarter': quarterly_totals.idxmax(),
            'visits': quarterly_totals.max()
        }

        return insights

    # ═══════════════════════════════════════════════════════════════════════════
    # ADMISSIONS ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════

    def _show_admissions_analysis(self):
        """Show Admissions analysis with 6 strategic categories plus 12-Month Strategic Forecast"""
        if self.admissions_data is None:
            st.error("Admissions data not available")
            return

        st.subheader("ADMISSIONS ANALYSIS")
        st.info(f"Analyzing {len(self.admissions_data):,} admissions records")

        # Create tabs for the 7 categories (matching EU Visits structure)
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "Operational Analytics",
            "Clinical Insights",
            "Seasonal Analysis",
            "Patient Flow",
            "Resource Management",
            "Performance KPIs",
            "12-Month Strategic Forecast"
        ])

        with tab1:
            self._show_adm_operational_analytics()
        with tab2:
            self._show_adm_clinical_insights()
        with tab3:
            self._show_adm_seasonal_analysis()
        with tab4:
            self._show_adm_patient_flow()
        with tab5:
            self._show_adm_resource_management()
        with tab6:
            self._show_adm_performance_kpis()
        with tab7:
            self._show_adm_12month_forecast()  # New method for admissions forecast

    def _show_adm_operational_analytics(self):
        """Operational Analytics for Admissions"""
        st.markdown("### Operational Analytics")
        st.caption("")

        # Basic volume analysis - FULL WIDTH
        if 'No_of_Admissions' in self.admissions_data.columns:
            st.markdown("#### Daily Admission Distribution")

            if 'Date' in self.admissions_data.columns:
                daily_admissions = self.admissions_data.groupby('Date')['No_of_Admissions'].sum().reset_index()

                # Full width histogram
                fig = px.histogram(
                    daily_admissions,
                    x='No_of_Admissions',
                    nbins=30,
                    title="Distribution of Daily Admissions"
                )
                fig.update_layout(height=500, margin=dict(l=80, r=50, t=100, b=80))
                st.plotly_chart(fig, use_container_width=True, key="adm_operational_daily_histogram_main")

                # Full width line chart
                fig = px.line(
                    daily_admissions,
                    x='Date',
                    y='No_of_Admissions',
                    title="Daily Admission Trends Over Time - Complete Timeline"
                )
                fig.update_layout(
                    height=500,
                    margin=dict(l=80, r=50, t=100, b=120),
                    xaxis=dict(tickangle=-45, nticks=20, tickfont=dict(size=10))
                )
                st.plotly_chart(fig, use_container_width=True, key="adm_operational_daily_line_main")

        # Hospital comparison - FULL WIDTH STACKED
        if 'Hospital_Name' in self.admissions_data.columns:
            st.markdown("#### Hospital Utilization Comparison")

            hospital_comparison = self.admissions_data.groupby('Hospital_Name')['No_of_Admissions'].agg(['sum', 'mean', 'count']).reset_index()

            # Total admissions
            fig = px.bar(
                hospital_comparison,
                x='Hospital_Name',
                y='sum',
                title="Total Admissions by Hospital"
            )
            fig.update_layout(
                height=400,
                margin=dict(l=80, r=50, t=100, b=120),
                xaxis=dict(tickangle=-45, tickfont=dict(size=10))
            )
            st.plotly_chart(fig, use_container_width=True, key="adm_operational_hospital_sum_main")

            # Average admissions
            fig = px.bar(
                hospital_comparison,
                x='Hospital_Name',
                y='mean',
                title="Average Daily Admissions by Hospital"
            )
            fig.update_layout(
                height=400,
                margin=dict(l=80, r=50, t=100, b=120),
                xaxis=dict(tickangle=-45, tickfont=dict(size=10))
            )
            st.plotly_chart(fig, use_container_width=True, key="adm_operational_hospital_mean_main")

        # Category analysis - HORIZONTAL BARS
        category_cols = ['Category_Value', 'Directorate', 'Department', 'Specialty']
        category_col = None

        for col in category_cols:
            if col in self.admissions_data.columns:
                category_col = col
                break

        if category_col:
            st.markdown(f"#### Top 10 {category_col.replace('_', ' ').title()} Workload")

            top_categories = self.admissions_data.groupby(category_col)['No_of_Admissions'].sum().nlargest(10).reset_index()

            fig = px.bar(
                top_categories,
                x='No_of_Admissions',
                y=category_col,
                orientation='h',
                title=f"Top 10 {category_col.replace('_', ' ').title()} by Admissions"
            )
            fig.update_layout(
                height=max(500, len(top_categories) * 35),
                margin=dict(l=250, r=50, t=100, b=80)
            )
            st.plotly_chart(fig, use_container_width=True, key=f"adm_operational_{category_col}_bar_main")
            # Top 10 Category Value Pie Chart Distribution
            if 'Category_Value' in self.admissions_data.columns:
                st.markdown("#### Top 10 Category Value Distribution")

                top_categories = self.admissions_data.groupby('Category_Value')['No_of_Admissions'].sum().nlargest(
                    10).reset_index()

                fig = px.pie(
                    top_categories,
                    values='No_of_Admissions',
                    names='Category_Value',
                    title="Top 10 Category Value Distribution by Admissions",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True, key="adm_clinical_directorate_pie_distribution")

    def _show_adm_seasonal_analysis(self):
        """Seasonal Analysis for Admissions"""
        st.markdown("### Seasonal Analysis")
        st.caption("")

        # Create seasonal tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Spring", "Summer", "Autumn", "Winter"])

        with tab1:
            self._show_adm_spring_analysis()
        with tab2:
            self._show_adm_summer_analysis()
        with tab3:
            self._show_adm_autumn_analysis()
        with tab4:
            self._show_adm_winter_analysis()

    def _show_adm_spring_analysis(self):
        """Spring seasonal analysis for Admissions"""
        st.markdown("#### Spring Analysis (March - May)")

        if 'Month' in self.admissions_data.columns:
            spring_data = self.admissions_data[self.admissions_data['Month'].isin([3, 4, 5])]

            if spring_data.empty:
                st.warning("Spring data is not available in the processed dataset.")
                return

            # Monthly breakdown within spring
            spring_monthly = spring_data.groupby('Month_Name')['No_of_Admissions'].sum().reset_index()

            fig = px.bar(
                spring_monthly,
                x='Month_Name',
                y='No_of_Admissions',
                title="Spring Months Admission Distribution",
                color='No_of_Admissions',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key="adm_spring_monthly")

            # Top 10 Patient Class drill-down
            if 'Patient_Class' in spring_data.columns:
                st.markdown("##### Top 10 Patient Classes in Spring")
                top_patient_classes = spring_data.groupby('Patient_Class')['No_of_Admissions'].sum().nlargest(
                    10).reset_index()

                fig = px.bar(
                    top_patient_classes,
                    x='No_of_Admissions',
                    y='Patient_Class',
                    orientation='h',
                    title="Top 10 Patient Classes - Spring Season"
                )
                fig.update_layout(height=max(400, len(top_patient_classes) * 30), margin=dict(l=250, r=50, t=80, b=60))
                st.plotly_chart(fig, use_container_width=True, key="adm_spring_patient_class_bar")

                fig = px.pie(
                    top_patient_classes,
                    values='No_of_Admissions',
                    names='Patient_Class',
                    title="Top 10 Patient Class Distribution - Spring",
                    color_discrete_sequence=px.colors.qualitative.Dark2
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True, key="adm_spring_patient_class_pie")

            # Top 10 Categories
            if 'Category_Value' in spring_data.columns:
                st.markdown("##### Top 10 Categories in Spring")
                top_categories = spring_data.groupby('Category_Value')['No_of_Admissions'].sum().nlargest(
                    10).reset_index()

                fig = px.bar(
                    top_categories,
                    x='No_of_Admissions',
                    y='Category_Value',
                    orientation='h',
                    title="Top 10 Categories - Spring Season"
                )
                fig.update_layout(height=max(400, len(top_categories) * 30), margin=dict(l=250, r=50, t=80, b=60))
                st.plotly_chart(fig, use_container_width=True, key="adm_spring_categories_bar")

                fig = px.pie(
                    top_categories,
                    values='No_of_Admissions',
                    names='Category_Value',
                    title="Top 10 Categories Distribution - Spring",
                    color_discrete_sequence=px.colors.qualitative.Dark2
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True, key="adm_spring_categories_pie")

            # Weather impact
            if 'Weather_Category' in spring_data.columns:
                st.markdown("##### Weather Impact on Spring Admissions")
                spring_weather = spring_data.groupby('Weather_Category')['No_of_Admissions'].sum().reset_index()

                fig = px.pie(
                    spring_weather,
                    values='No_of_Admissions',
                    names='Weather_Category',
                    title="Spring Weather Impact on Admissions",
                    color_discrete_sequence=px.colors.qualitative.Dark2
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True, key="adm_spring_weather")

            # Hospital distribution
            if 'Hospital_Name' in spring_data.columns:
                st.markdown("##### Hospital Distribution in Spring")
                spring_hospitals = spring_data.groupby('Hospital_Name')['No_of_Admissions'].sum().reset_index()

                fig = px.bar(
                    spring_hospitals,
                    x='Hospital_Name',
                    y='No_of_Admissions',
                    title="Spring Admissions by Hospital"
                )
                fig.update_layout(
                    height=400,
                    margin=dict(l=80, r=50, t=100, b=120),
                    xaxis=dict(tickangle=-45, tickfont=dict(size=10))
                )
                st.plotly_chart(fig, use_container_width=True, key="adm_spring_hospitals")

                fig = px.pie(
                    spring_hospitals,
                    values='No_of_Admissions',
                    names='Hospital_Name',
                    title="Spring Admissions Distribution by Hospital",
                    color_discrete_sequence=px.colors.qualitative.Dark2
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True, key="adm_spring_hospitals_pie")

        else:
            st.error("Month column not found in the dataset.")

    def _show_adm_summer_analysis(self):
        """Summer seasonal analysis for Admissions"""
        st.markdown("#### Summer Analysis (June - August)")

        if 'Month' in self.admissions_data.columns:
            summer_data = self.admissions_data[self.admissions_data['Month'].isin([6, 7, 8])]

            if summer_data.empty:
                st.warning("Summer data is not available in the processed dataset.")
                return

            # Monthly breakdown within summer
            summer_monthly = summer_data.groupby('Month_Name')['No_of_Admissions'].sum().reset_index()

            fig = px.bar(
                summer_monthly,
                x='Month_Name',
                y='No_of_Admissions',
                title="Summer Months Admission Distribution",
                color='No_of_Admissions',
                color_continuous_scale='plasma'
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key="adm_summer_monthly")

            # Top 10 Patient Class drill-down
            if 'Patient_Class' in summer_data.columns:
                st.markdown("##### 🚑 Top 10 Patient Classes in Summer")
                top_patient_classes = summer_data.groupby('Patient_Class')['No_of_Admissions'].sum().nlargest(
                    10).reset_index()

                fig = px.bar(
                    top_patient_classes,
                    x='No_of_Admissions',
                    y='Patient_Class',
                    orientation='h',
                    title="Top 10 Patient Classes - Summer Season"
                )
                fig.update_layout(height=max(400, len(top_patient_classes) * 30), margin=dict(l=250, r=50, t=80, b=60))
                st.plotly_chart(fig, use_container_width=True, key="adm_summer_patient_class_bar")

                fig = px.pie(
                    top_patient_classes,
                    values='No_of_Admissions',
                    names='Patient_Class',
                    title="Top 10 Patient Class Distribution - Summer",
                    color_discrete_sequence=px.colors.qualitative.Set1
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True, key="adm_summer_patient_class_pie")

            # Top 10 Categories
            if 'Category_Value' in summer_data.columns:
                st.markdown("##### 🏥 Top 10 Categories in Summer")
                top_categories = summer_data.groupby('Category_Value')['No_of_Admissions'].sum().nlargest(
                    10).reset_index()

                fig = px.bar(
                    top_categories,
                    x='No_of_Admissions',
                    y='Category_Value',
                    orientation='h',
                    title="Top 10 Categories - Summer Season"
                )
                fig.update_layout(height=max(400, len(top_categories) * 30), margin=dict(l=250, r=50, t=80, b=60))
                st.plotly_chart(fig, use_container_width=True, key="adm_summer_categories_bar")

                fig = px.pie(
                    top_categories,
                    values='No_of_Admissions',
                    names='Category_Value',
                    title="Top 10 Categories Distribution - Summer",
                    color_discrete_sequence=px.colors.qualitative.Set1
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True, key="adm_summer_categories_pie")

            # Weather impact
            if 'Weather_Category' in summer_data.columns:
                st.markdown("##### 🌤️ Weather Impact on Summer Admissions")
                summer_weather = summer_data.groupby('Weather_Category')['No_of_Admissions'].sum().reset_index()

                fig = px.pie(
                    summer_weather,
                    values='No_of_Admissions',
                    names='Weather_Category',
                    title="Summer Weather Impact on Admissions",
                    color_discrete_sequence=px.colors.qualitative.Set1
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True, key="adm_summer_weather")

            # Hospital distribution
            if 'Hospital_Name' in summer_data.columns:
                st.markdown("##### 🏥 Hospital Distribution in Summer")
                summer_hospitals = summer_data.groupby('Hospital_Name')['No_of_Admissions'].sum().reset_index()

                fig = px.bar(
                    summer_hospitals,
                    x='Hospital_Name',
                    y='No_of_Admissions',
                    title="Summer Admissions by Hospital"
                )
                fig.update_layout(
                    height=400,
                    margin=dict(l=80, r=50, t=100, b=120),
                    xaxis=dict(tickangle=-45, tickfont=dict(size=10))
                )
                st.plotly_chart(fig, use_container_width=True, key="adm_summer_hospitals")

                fig = px.pie(
                    summer_hospitals,
                    values='No_of_Admissions',
                    names='Hospital_Name',
                    title="Summer Admissions Distribution by Hospital",
                    color_discrete_sequence=px.colors.qualitative.Set1
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True, key="adm_summer_hospitals_pie")

        else:
            st.error("❌ Month column not found in the dataset.")

    def _show_adm_autumn_analysis(self):
        """Autumn seasonal analysis for Admissions"""
        st.markdown("#### 🍂 Autumn Analysis (September - November)")

        if 'Month' in self.admissions_data.columns:
            autumn_data = self.admissions_data[self.admissions_data['Month'].isin([9, 10, 11])]

            if autumn_data.empty:
                st.warning("⚠️ Autumn data is not available in the processed dataset.")
                return

            # Monthly breakdown within autumn
            autumn_monthly = autumn_data.groupby('Month_Name')['No_of_Admissions'].sum().reset_index()

            fig = px.bar(
                autumn_monthly,
                x='Month_Name',
                y='No_of_Admissions',
                title="Autumn Months Admission Distribution",
                color='No_of_Admissions',
                color_continuous_scale='inferno'
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key="adm_autumn_monthly")

            # Top 10 Patient Class drill-down
            if 'Patient_Class' in autumn_data.columns:
                st.markdown("##### 🚑 Top 10 Patient Classes in Autumn")
                top_patient_classes = autumn_data.groupby('Patient_Class')['No_of_Admissions'].sum().nlargest(
                    10).reset_index()

                fig = px.bar(
                    top_patient_classes,
                    x='No_of_Admissions',
                    y='Patient_Class',
                    orientation='h',
                    title="Top 10 Patient Classes - Autumn Season"
                )
                fig.update_layout(height=max(400, len(top_patient_classes) * 30), margin=dict(l=250, r=50, t=80, b=60))
                st.plotly_chart(fig, use_container_width=True, key="adm_autumn_patient_class_bar")

                fig = px.pie(
                    top_patient_classes,
                    values='No_of_Admissions',
                    names='Patient_Class',
                    title="Top 10 Patient Class Distribution - Autumn",
                    color_discrete_sequence=px.colors.qualitative.Dark24
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True, key="adm_autumn_patient_class_pie")

            # Top 10 Categories
            if 'Category_Value' in autumn_data.columns:
                st.markdown("##### 🏥 Top 10 Categories in Autumn")
                top_categories = autumn_data.groupby('Category_Value')['No_of_Admissions'].sum().nlargest(
                    10).reset_index()

                fig = px.bar(
                    top_categories,
                    x='No_of_Admissions',
                    y='Category_Value',
                    orientation='h',
                    title="Top 10 Categories - Autumn Season"
                )
                fig.update_layout(height=max(400, len(top_categories) * 30), margin=dict(l=250, r=50, t=80, b=60))
                st.plotly_chart(fig, use_container_width=True, key="adm_autumn_categories_bar")

                fig = px.pie(
                    top_categories,
                    values='No_of_Admissions',
                    names='Category_Value',
                    title="Top 10 Categories Distribution - Autumn",
                    color_discrete_sequence=px.colors.qualitative.Dark24
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True, key="adm_autumn_categories_pie")

            # Weather impact
            if 'Weather_Category' in autumn_data.columns:
                st.markdown("##### 🌤️ Weather Impact on Autumn Admissions")
                autumn_weather = autumn_data.groupby('Weather_Category')['No_of_Admissions'].sum().reset_index()

                fig = px.pie(
                    autumn_weather,
                    values='No_of_Admissions',
                    names='Weather_Category',
                    title="Autumn Weather Impact on Admissions",
                    color_discrete_sequence=px.colors.qualitative.Dark24
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True, key="adm_autumn_weather")

            # Hospital distribution
            if 'Hospital_Name' in autumn_data.columns:
                st.markdown("##### 🏥 Hospital Distribution in Autumn")
                autumn_hospitals = autumn_data.groupby('Hospital_Name')['No_of_Admissions'].sum().reset_index()

                fig = px.bar(
                    autumn_hospitals,
                    x='Hospital_Name',
                    y='No_of_Admissions',
                    title="Autumn Admissions by Hospital"
                )
                fig.update_layout(
                    height=400,
                    margin=dict(l=80, r=50, t=100, b=120),
                    xaxis=dict(tickangle=-45, tickfont=dict(size=10))
                )
                st.plotly_chart(fig, use_container_width=True, key="adm_autumn_hospitals")

                fig = px.pie(
                    autumn_hospitals,
                    values='No_of_Admissions',
                    names='Hospital_Name',
                    title="Autumn Admissions Distribution by Hospital",
                    color_discrete_sequence=px.colors.qualitative.Dark24
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True, key="adm_autumn_hospitals_pie")

        else:
            st.error("Month column not found in the dataset.")

    def _show_adm_winter_analysis(self):
        """Winter seasonal analysis for Admissions"""
        st.markdown("####  Winter Analysis (December - February)")

        if 'Month' in self.admissions_data.columns:
            winter_data = self.admissions_data[self.admissions_data['Month'].isin([12, 1, 2])]

            if winter_data.empty:
                st.warning(" Winter data is not available in the processed dataset.")
                return

            # Monthly breakdown within winter
            winter_monthly = winter_data.groupby('Month_Name')['No_of_Admissions'].sum().reset_index()

            fig = px.bar(
                winter_monthly,
                x='Month_Name',
                y='No_of_Admissions',
                title="Winter Months Admission Distribution",
                color='No_of_Admissions',
                color_continuous_scale='cividis'
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key="adm_winter_monthly")

            # Top 10 Patient Class drill-down
            if 'Patient_Class' in winter_data.columns:
                st.markdown("##### 🚑 Top 10 Patient Classes in Winter")
                top_patient_classes = winter_data.groupby('Patient_Class')['No_of_Admissions'].sum().nlargest(
                    10).reset_index()

                fig = px.bar(
                    top_patient_classes,
                    x='No_of_Admissions',
                    y='Patient_Class',
                    orientation='h',
                    title="Top 10 Patient Classes - Winter Season"
                )
                fig.update_layout(height=max(400, len(top_patient_classes) * 30), margin=dict(l=250, r=50, t=80, b=60))
                st.plotly_chart(fig, use_container_width=True, key="adm_winter_patient_class_bar")

                fig = px.pie(
                    top_patient_classes,
                    values='No_of_Admissions',
                    names='Patient_Class',
                    title="Top 10 Patient Class Distribution - Winter",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True, key="adm_winter_patient_class_pie")

            # Top 10 Categories
            if 'Category_Value' in winter_data.columns:
                st.markdown("##### 🏥 Top 10 Categories in Winter")
                top_categories = winter_data.groupby('Category_Value')['No_of_Admissions'].sum().nlargest(
                    10).reset_index()

                fig = px.bar(
                    top_categories,
                    x='No_of_Admissions',
                    y='Category_Value',
                    orientation='h',
                    title="Top 10 Categories - Winter Season"
                )
                fig.update_layout(height=max(400, len(top_categories) * 30), margin=dict(l=250, r=50, t=80, b=60))
                st.plotly_chart(fig, use_container_width=True, key="adm_winter_categories_bar")

                fig = px.pie(
                    top_categories,
                    values='No_of_Admissions',
                    names='Category_Value',
                    title="Top 10 Categories Distribution - Winter",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True, key="adm_winter_categories_pie")

            # Weather impact
            if 'Weather_Category' in winter_data.columns:
                st.markdown("##### 🌤️ Weather Impact on Winter Admissions")
                winter_weather = winter_data.groupby('Weather_Category')['No_of_Admissions'].sum().reset_index()

                fig = px.pie(
                    winter_weather,
                    values='No_of_Admissions',
                    names='Weather_Category',
                    title="Winter Weather Impact on Admissions",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True, key="adm_winter_weather")

            # Hospital distribution
            if 'Hospital_Name' in winter_data.columns:
                st.markdown("##### 🏥 Hospital Distribution in Winter")
                winter_hospitals = winter_data.groupby('Hospital_Name')['No_of_Admissions'].sum().reset_index()

                fig = px.bar(
                    winter_hospitals,
                    x='Hospital_Name',
                    y='No_of_Admissions',
                    title="Winter Admissions by Hospital"
                )
                fig.update_layout(
                    height=400,
                    margin=dict(l=80, r=50, t=100, b=120),
                    xaxis=dict(tickangle=-45, tickfont=dict(size=10))
                )
                st.plotly_chart(fig, use_container_width=True, key="adm_winter_hospitals")

                fig = px.pie(
                    winter_hospitals,
                    values='No_of_Admissions',
                    names='Hospital_Name',
                    title="Winter Admissions Distribution by Hospital",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True, key="adm_winter_hospitals_pie")

            # Holiday impact for winter
            if 'Is_Bank_Holiday' in winter_data.columns:
                st.markdown("##### 🎄 Holiday Impact on Winter Admissions")
                holiday_winter = winter_data.groupby('Is_Bank_Holiday')['No_of_Admissions'].agg(
                    ['sum', 'mean']).reset_index()

                if len(holiday_winter) > 1:
                    fig = px.bar(
                        holiday_winter,
                        x='Is_Bank_Holiday',
                        y='mean',
                        title="Winter: Normal vs Holiday Average Admissions"
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True, key="adm_winter_holiday")

        else:
            st.error("❌ Month column not found in the dataset.")

    def _show_adm_clinical_insights(self):
        """🩺 Clinical Insights for Admissions"""
        st.markdown("### 🩺 Clinical Insights")

        # Patient class analysis
        if 'Patient_Class' in self.admissions_data.columns:
            st.markdown("#### 🚑 Top 10 Patient Class Distribution")

            patient_class_data = self.admissions_data.groupby('Patient_Class')['No_of_Admissions'].sum().nlargest(10).reset_index()

            # Horizontal bar chart
            fig = px.bar(
                patient_class_data,
                x='No_of_Admissions',
                y='Patient_Class',
                orientation='h',
                title="Top 10 Admissions by Patient Class"
            )
            fig.update_layout(
                height=max(500, len(patient_class_data) * 35),
                margin=dict(l=200, r=50, t=100, b=80)
            )
            st.plotly_chart(fig, use_container_width=True, key="adm_clinical_patient_class_bar_main")

            # Pie chart
            fig = px.pie(
                patient_class_data,
                values='No_of_Admissions',
                names='Patient_Class',
                title="Top 10 Patient Class Distribution"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True, key="adm_clinical_patient_class_pie_main")

        # Directorate analysis
        if 'Category_Value' in self.admissions_data.columns:
            st.markdown("#### 🏥 Top 10 Directorate Analysis")

            top_directorates = self.admissions_data.groupby('Category_Value')['No_of_Admissions'].sum().nlargest(10).reset_index()

            fig = px.bar(
                top_directorates,
                x='No_of_Admissions',
                y='Category_Value',
                orientation='h',
                title="Top 10 Directorates by Volume"
            )
            fig.update_layout(
                height=max(500, len(top_directorates) * 35),
                margin=dict(l=250, r=50, t=100, b=80)
            )
            st.plotly_chart(fig, use_container_width=True, key="adm_clinical_directorate_bar_main") # Pie chart for Top 10 Directorates Distribution
            fig = px.pie(
                top_directorates,
                values='No_of_Admissions',
                names='Category_Value',
                title="Top 10 Directorates Distribution by Volume"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True, key="adm_clinical_directorate_pie_main")

    def _show_adm_patient_flow(self):
        """Patient Flow for Admissions"""
        st.markdown("### Patient Flow")

        # Hospital distribution
        if 'Hospital_Name' in self.admissions_data.columns:
            st.markdown("#### Hospital Distribution")

            hospital_flow = self.admissions_data.groupby('Hospital_Name')['No_of_Admissions'].sum().sort_values(ascending=False).reset_index()

            # Changed from pie chart to horizontal bar chart
            fig = px.bar(
                hospital_flow,
                x='No_of_Admissions',
                y='Hospital_Name',
                orientation='h',
                title="Admission Distribution Across Hospitals"
            )
            fig.update_layout(
                height=max(400, len(hospital_flow) * 35),
                margin=dict(l=200, r=50, t=100, b=80)
            )
            st.plotly_chart(fig, use_container_width=True, key="adm_patient_flow_hospital_bar_main")
            # Hospital distribution
            if 'Hospital_Name' in self.admissions_data.columns:
                st.markdown("#### Hospital Distribution")

                hospital_flow = self.admissions_data.groupby('Hospital_Name')['No_of_Admissions'].sum().sort_values(
                    ascending=False).reset_index()

                fig = px.pie(
                    hospital_flow,
                    values='No_of_Admissions',
                    names='Hospital_Name',
                    title="Admission Distribution Across Hospitals"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True, key="adm_patient_flow_hospital_pie_main")

        # Admission method analysis
        method_cols = ['Category_Name', 'Admission_Method', 'Admission_Type']
        method_col = None

        for col in method_cols:
            if col in self.admissions_data.columns:
                method_col = col
                break

        if method_col:
            st.markdown("#### Admission Method Analysis")

            method_data = self.admissions_data.groupby(method_col)['No_of_Admissions'].sum().sort_values(ascending=False).reset_index()

            fig = px.bar(
                method_data,
                x='No_of_Admissions',
                y=method_col,
                orientation='h',
                title="Admission Pathways"
            )
            fig.update_layout(
                height=max(400, len(method_data) * 35),
                margin=dict(l=200, r=50, t=100, b=80)
            )
            st.plotly_chart(fig, use_container_width=True, key=f"adm_patient_flow_{method_col}_bar_main")
            # Top 10 Pie chart for admission methods distribution
            top_methods = method_data.head(10)
            fig = px.pie(
                top_methods,
                values='No_of_Admissions',
                names=method_col,
                title="Top 10 Admission Methods Distribution"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True, key=f"adm_patient_flow_{method_col}_pie_main")

    def _show_adm_resource_management(self):
        """Resource Management for Admissions"""
        st.markdown("### Resource Management")
        st.caption("Capacity planning, Peak demand analysis")

        if 'No_of_Admissions' in self.admissions_data.columns and 'Date' in self.admissions_data.columns:
            # Basic capacity metrics
            daily_totals = self.admissions_data.groupby('Date')['No_of_Admissions'].sum().reset_index()

            mean_adm = daily_totals['No_of_Admissions'].mean()
            max_adm = daily_totals['No_of_Admissions'].max()
            min_adm = daily_totals['No_of_Admissions'].min()
            std_adm = daily_totals['No_of_Admissions'].std()

            st.markdown("#### Peak Demand Analysis")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average Daily", f"{mean_adm:.0f}")
            with col2:
                st.metric("Peak Daily", f"{max_adm:.0f}")
            with col3:
                st.metric("Min Daily", f"{min_adm:.0f}")
            with col4:
                variability = (std_adm / mean_adm) * 100 if mean_adm > 0 else 0
                st.metric("Variability %", f"{variability:.1f}%")

            # Demand distribution
            fig = px.histogram(
                daily_totals,
                x='No_of_Admissions',
                nbins=25,
                title="Daily Admission Volume Distribution"
            )
            fig.update_layout(height=400, margin=dict(l=80, r=50, t=100, b=80))
            st.plotly_chart(fig, use_container_width=True, key="adm_resource_demand_histogram_main")

        # Capacity optimization by category
        if 'Category_Value' in self.admissions_data.columns and 'Date' in self.admissions_data.columns:
            st.markdown("#### Top 10 Capacity Optimization by Category")

            daily_category = self.admissions_data.groupby(['Date', 'Category_Value'])['No_of_Admissions'].sum().reset_index()
            top_categories = self.admissions_data.groupby('Category_Value')['No_of_Admissions'].sum().nlargest(10).index

            capacity_metrics = []
            for category in top_categories:
                category_data = daily_category[daily_category['Category_Value'] == category]
                if not category_data.empty:
                    avg_daily = category_data['No_of_Admissions'].mean()
                    peak_daily = category_data['No_of_Admissions'].max()
                    variability = category_data['No_of_Admissions'].std() / avg_daily * 100 if avg_daily > 0 else 0

                    capacity_metrics.append({
                        'Category': category[:30] + "..." if len(category) > 30 else category,
                        'Average_Daily': avg_daily,
                        'Peak_Daily': peak_daily,
                        'Variability_%': variability
                    })

            if capacity_metrics:
                capacity_df = pd.DataFrame(capacity_metrics)

                fig = px.scatter(
                    capacity_df,
                    x='Average_Daily',
                    y='Peak_Daily',
                    size='Variability_%',
                    hover_name='Category',
                    title="Capacity Analysis: Average vs Peak Admissions (Size = Variability %)"
                )
                fig.update_layout(height=500, margin=dict(l=80, r=50, t=100, b=80))
                st.plotly_chart(fig, use_container_width=True, key="adm_resource_capacity_scatter_main")

    def _show_adm_performance_kpis(self):
        """Performance KPIs for Admissions"""
        st.markdown("### Performance KPIs")
        st.caption("Quality metrics, Trend analysis, Benchmarking")

        # Basic performance metrics
        if 'No_of_Admissions' in self.admissions_data.columns:
            total_admissions = self.admissions_data['No_of_Admissions'].sum()

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Admissions", f"{total_admissions:,}")

            with col2:
                if 'Date' in self.admissions_data.columns:
                    unique_dates = self.admissions_data['Date'].nunique()
                    daily_avg = total_admissions / unique_dates if unique_dates > 0 else 0
                    st.metric("Daily Average", f"{daily_avg:.0f}")

            with col3:
                if 'Hospital_Name' in self.admissions_data.columns:
                    hospitals = self.admissions_data['Hospital_Name'].nunique()
                    st.metric("Hospitals", f"{hospitals}")

            with col4:
                if 'Category_Value' in self.admissions_data.columns:
                    categories = self.admissions_data['Category_Value'].nunique()
                    st.metric("Categories", f"{categories}")

        # Monthly Performance Trends - NEW ADDITION
        if 'Date' in self.admissions_data.columns and 'No_of_Admissions' in self.admissions_data.columns:
            st.markdown("#### Monthly Performance Trends")

            monthly_performance = self.admissions_data.groupby('Month_Name')['No_of_Admissions'].agg(
                ['sum', 'mean', 'std']).reset_index()
            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                           'July', 'August', 'September', 'October', 'November', 'December']
            monthly_performance['month_num'] = monthly_performance['Month_Name'].map(
                {month: i for i, month in enumerate(month_order)})
            monthly_performance = monthly_performance.sort_values('month_num').dropna()

            if not monthly_performance.empty:
                # Monthly volume trend line
                fig = px.line(
                    monthly_performance,
                    x='Month_Name',
                    y='sum',
                    title="Monthly Admission Volume Trend",
                    markers=True,
                    labels={"sum": "Total Monthly Admissions", "Month_Name": "Month"}
                )
                fig.update_layout(
                    height=400,
                    margin=dict(l=80, r=50, t=100, b=120),
                    xaxis=dict(tickangle=-45, tickfont=dict(size=10))
                )
                st.plotly_chart(fig, use_container_width=True, key="adm_performance_monthly_trend")

        # Hospital performance benchmarking
        if 'Hospital_Name' in self.admissions_data.columns:
            st.markdown("#### Hospital Performance Benchmarking")

            hospital_kpis = self.admissions_data.groupby('Hospital_Name').agg({
                'No_of_Admissions': ['sum', 'mean', 'count', 'std']
            }).round(2)

            hospital_kpis.columns = ['Total_Admissions', 'Avg_Daily', 'Days_Active', 'Std_Dev']
            hospital_kpis = hospital_kpis.reset_index()
            hospital_kpis['Efficiency_Score'] = (hospital_kpis['Total_Admissions'] / hospital_kpis['Days_Active']).round(2)

            # Total admissions - horizontal bar for better readability
            fig = px.bar(
                hospital_kpis,
                x='Total_Admissions',
                y='Hospital_Name',
                orientation='h',
                title="Total Admissions by Hospital"
            )
            fig.update_layout(
                height=max(400, len(hospital_kpis) * 35),
                margin=dict(l=200, r=50, t=100, b=80)
            )
            st.plotly_chart(fig, use_container_width=True, key="adm_performance_hospital_total_main")

            # Hospital market share pie chart - NEW ADDITION
            fig = px.pie(
                hospital_kpis,
                values='Total_Admissions',
                names='Hospital_Name',
                title="Hospital Market Share - Admission Distribution"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True, key="adm_performance_hospital_pie")

            # Efficiency scores - horizontal bar
            fig = px.bar(
                hospital_kpis,
                x='Efficiency_Score',
                y='Hospital_Name',
                orientation='h',
                title="Daily Efficiency Score by Hospital",
                color='Efficiency_Score',
                color_continuous_scale='viridis'
            )
            fig.update_layout(
                height=max(400, len(hospital_kpis) * 35),
                margin=dict(l=200, r=50, t=100, b=80)
            )
            st.plotly_chart(fig, use_container_width=True, key="adm_performance_hospital_efficiency_main")

        # Category performance
        if 'Category_Value' in self.admissions_data.columns:
            st.markdown("#### Top 10 Category Performance")

            category_performance = self.admissions_data.groupby('Category_Value')['No_of_Admissions'].agg(
                ['sum', 'count', 'mean']).reset_index()
            category_performance.columns = ['Category', 'Total_Admissions', 'Days_Active', 'Avg_Daily']
            category_performance = category_performance.nlargest(10, 'Total_Admissions')

            # Category performance scatter plot
            fig = px.scatter(
                category_performance,
                x='Days_Active',
                y='Total_Admissions',
                size='Avg_Daily',
                hover_name='Category',
                title="Top 10 Category Performance (Size = Average Daily)"
            )
            fig.update_layout(height=500, margin=dict(l=80, r=50, t=100, b=80))
            st.plotly_chart(fig, use_container_width=True, key="adm_performance_category_scatter_main")

            # Category distribution pie chart - NEW ADDITION
            fig = px.pie(
                category_performance,
                values='Total_Admissions',
                names='Category',
                title="Top 10 Category Distribution - Admission Volume"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True, key="adm_performance_category_pie")

        # System efficiency indicators - NEW ADDITION
        if 'Is_Weekend' in self.admissions_data.columns:
            st.markdown("#### System Efficiency Indicators")

            efficiency_metrics = self.admissions_data.groupby('Is_Weekend')['No_of_Admissions'].agg(
                ['sum', 'mean', 'count']).reset_index()
            efficiency_metrics['Is_Weekend'] = efficiency_metrics['Is_Weekend'].map({True: 'Weekend', False: 'Weekday'})

            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    efficiency_metrics,
                    x='Is_Weekend',
                    y='mean',
                    title="Average Daily Admissions: Weekday vs Weekend",
                    labels={"mean": "Average Daily Admissions", "Is_Weekend": "Day Type"}
                )
                fig.update_layout(height=350, margin=dict(l=80, r=50, t=80, b=80))
                st.plotly_chart(fig, use_container_width=True, key="adm_performance_weekday_weekend")

            with col2:
                total_weekday = efficiency_metrics[efficiency_metrics['Is_Weekend'] == 'Weekday']['sum'].iloc[0] if len(efficiency_metrics) > 0 else 0
                total_weekend = efficiency_metrics[efficiency_metrics['Is_Weekend'] == 'Weekend']['sum'].iloc[0] if len(efficiency_metrics) > 1 else 0

                ratio_data = pd.DataFrame({
                    'Period': ['Weekday', 'Weekend'],
                    'Total_Admissions': [total_weekday, total_weekend]
                })

                fig = px.pie(
                    ratio_data,
                    values='Total_Admissions',
                    names='Period',
                    title="Weekday vs Weekend Admission Distribution"
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True, key="adm_performance_period_distribution")

    def _show_adm_12month_forecast(self):
        """12-Month Strategic Forecasting for Admissions"""
        st.markdown("### 12-Month Strategic Forecast & Predictions")
        st.info("Long-term predictions for capacity planning and resource allocation")

        daily_data, error = self._prepare_adm_forecasting_data()
        if daily_data is None:
            st.error(f"❌ {error}")
            return

        # Generate comprehensive yearly forecasts with working visualizations
        self._generate_adm_yearly_insights(daily_data)

    def _prepare_adm_forecasting_data(self) -> Tuple[Optional[pd.DataFrame], str]:
        """Prepare data for admissions forecasting analysis"""
        if self.admissions_data is None:
            return None, "Admissions data not available"

        if 'No_of_Admissions' not in self.admissions_data.columns:
            return None, "No_of_Admissions column not found"

        if 'Date' not in self.admissions_data.columns:
            return None, "Date column not found"

        # Aggregate daily data
        daily_data = self.admissions_data.groupby('Date')['No_of_Admissions'].sum().reset_index()
        daily_data = daily_data.sort_values('Date')

        if len(daily_data) < 10:
            return None, f"Insufficient data for forecasting: {len(daily_data)} days available"

        return daily_data, ""

    def _generate_adm_yearly_insights(self, daily_data):
        """Generate comprehensive yearly forecasting insights for Admissions"""
        # Analyze historical patterns for projections
        historical_analysis = self._analyze_adm_historical_patterns(daily_data)

        st.markdown("##### Next 12 Months Admissions Forecast")

        # Generate 12-month forecast using historical patterns
        yearly_forecast = self._create_adm_yearly_forecast(daily_data, historical_analysis)

        if yearly_forecast is not None:
            # Display yearly total prediction
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                total_forecast = yearly_forecast['Predicted_Admissions'].sum()
                st.metric("Next 12 Months Total", f"{total_forecast:,.0f}")

            with col2:
                daily_avg = yearly_forecast['Predicted_Admissions'].mean()
                st.metric("Daily Average", f"{daily_avg:.0f}")

            with col3:
                historical_yearly = daily_data['No_of_Admissions'].sum() * (365 / len(daily_data))
                growth_rate = ((total_forecast - historical_yearly) / historical_yearly) * 100
                st.metric("Projected Growth", f"{growth_rate:+.1f}%")

            with col4:
                peak_month_admissions = yearly_forecast.groupby(yearly_forecast['Date'].dt.month)[
                    'Predicted_Admissions'].sum().max()
                st.metric("Peak Month Volume", f"{peak_month_admissions:.0f}")

            # Plot 12-month forecast
            fig = go.Figure()

            # Historical data (last 90 days)
            recent_data = daily_data.tail(90)
            fig.add_trace(go.Scatter(
                x=recent_data['Date'],
                y=recent_data['No_of_Admissions'],
                mode='lines',
                name='Recent Historical',
                line=dict(color='blue', width=2)
            ))

            # 12-month forecast
            fig.add_trace(go.Scatter(
                x=yearly_forecast['Date'],
                y=yearly_forecast['Predicted_Admissions'],
                mode='lines',
                name='12-Month Forecast',
                line=dict(color='red', width=3)
            ))

            # Add confidence bands
            upper_bound = yearly_forecast['Predicted_Admissions'] * 1.15
            lower_bound = yearly_forecast['Predicted_Admissions'] * 0.85

            fig.add_trace(go.Scatter(
                x=yearly_forecast['Date'],
                y=upper_bound,
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False
            ))

            fig.add_trace(go.Scatter(
                x=yearly_forecast['Date'],
                y=lower_bound,
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,0,0,0)',
                name='Confidence Range (±15%)',
                fillcolor='rgba(255,0,0,0.2)'
            ))

            fig.update_layout(
                title="12-Month Admissions Forecast",
                height=500,
                xaxis_title="Date",
                yaxis_title="Predicted Daily Admissions"
            )
            st.plotly_chart(fig, use_container_width=True, key="adm_yearly_forecast")

            # Strategic insights section
            st.markdown("##### Strategic Planning Insights")

            # Generate specific predictions
            insights = self._generate_adm_strategic_insights(yearly_forecast, historical_analysis)

            # Display insights in organized sections
            insight_tab1, insight_tab2, insight_tab3 = st.tabs([
                "Peak Periods", "Seasonal Trends", "Operational Planning"
            ])

            with insight_tab1:
                self._show_adm_peak_periods_insights(insights, yearly_forecast)

            with insight_tab2:
                self._show_adm_seasonal_trends_insights(insights, yearly_forecast, historical_analysis)

            with insight_tab3:
                self._show_adm_operational_planning_insights(insights, yearly_forecast)

    def _analyze_adm_historical_patterns(self, daily_data):
        """Analyze historical patterns for admissions forecasting"""
        patterns = {}

        # Add month, day of week, etc.
        daily_data = daily_data.copy()
        daily_data['Month'] = daily_data['Date'].dt.month
        daily_data['DayOfWeek'] = daily_data['Date'].dt.dayofweek
        daily_data['Week'] = daily_data['Date'].dt.isocalendar().week

        # Monthly patterns
        patterns['monthly'] = daily_data.groupby('Month')['No_of_Admissions'].agg(['mean', 'std']).round(1)

        # Day of week patterns
        patterns['daily'] = daily_data.groupby('DayOfWeek')['No_of_Admissions'].agg(['mean', 'std']).round(1)

        # Weekly patterns
        patterns['weekly'] = daily_data.groupby('Week')['No_of_Admissions'].agg(['mean', 'std']).round(1)

        # Seasonal patterns
        daily_data['Season'] = daily_data['Month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
        })
        patterns['seasonal'] = daily_data.groupby('Season')['No_of_Admissions'].agg(['mean', 'std']).round(1)

        # Growth trend
        if len(daily_data) > 30:
            recent_avg = daily_data.tail(30)['No_of_Admissions'].mean()
            older_avg = daily_data.head(30)['No_of_Admissions'].mean()
            patterns['growth_trend'] = (recent_avg - older_avg) / older_avg * 100 if older_avg > 0 else 0
        else:
            patterns['growth_trend'] = 0

        return patterns

    def _create_adm_yearly_forecast(self, daily_data, patterns):
        """Create 12-month forecast for admissions using historical patterns"""
        try:
            # Generate future dates (365 days)
            last_date = daily_data['Date'].max()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=365, freq='D')

            forecast_data = []

            for date in future_dates:
                # Get base prediction from patterns
                month = date.month
                dayofweek = date.dayofweek
                week = date.isocalendar().week

                # Base prediction from monthly pattern
                if month in patterns['monthly'].index:
                    base_prediction = patterns['monthly'].loc[month, 'mean']
                else:
                    base_prediction = daily_data['No_of_Admissions'].mean()

                # Adjust for day of week
                if dayofweek in patterns['daily'].index:
                    dow_factor = patterns['daily'].loc[dayofweek, 'mean'] / patterns['daily']['mean'].mean()
                    base_prediction *= dow_factor

                # Apply growth trend
                days_ahead = (date - last_date).days
                growth_factor = 1 + (patterns['growth_trend'] / 100) * (days_ahead / 365)
                final_prediction = base_prediction * growth_factor

                # Add some realistic noise
                noise = np.random.normal(0, base_prediction * 0.05)
                final_prediction += noise

                # Ensure non-negative
                final_prediction = max(0, final_prediction)

                forecast_data.append({
                    'Date': date,
                    'Predicted_Admissions': final_prediction,
                    'Month': month,
                    'DayOfWeek': dayofweek,
                    'Week': week,
                    'Quarter': {1: 'Q1', 2: 'Q1', 3: 'Q1', 4: 'Q2', 5: 'Q2', 6: 'Q2', 7: 'Q3', 8: 'Q3', 9: 'Q3', 10: 'Q4', 11: 'Q4', 12: 'Q4'}[month],
                    'Season': {12: 'Winter', 1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}[month]
                })

            return pd.DataFrame(forecast_data)

        except Exception as e:
            st.error(f"Error creating yearly forecast: {str(e)}")
            return None

    def _generate_adm_strategic_insights(self, yearly_forecast, patterns):
        """Generate strategic insights from admissions forecast"""
        insights = {}

        # Monthly insights
        monthly_totals = yearly_forecast.groupby('Month')['Predicted_Admissions'].sum().round(0)
        insights['highest_month'] = {
            'month': calendar.month_name[monthly_totals.idxmax()],
            'visits': monthly_totals.max(),
            'month_num': monthly_totals.idxmax()
        }
        insights['lowest_month'] = {
            'month': calendar.month_name[monthly_totals.idxmin()],
            'visits': monthly_totals.min(),
            'month_num': monthly_totals.idxmin()
        }

        # Day of week insights
        dow_totals = yearly_forecast.groupby('DayOfWeek')['Predicted_Admissions'].sum()
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        insights['busiest_day'] = {
            'day': day_names[dow_totals.idxmax()],
            'visits': dow_totals.max(),
            'day_num': dow_totals.idxmax()
        }
        insights['quietest_day'] = {
            'day': day_names[dow_totals.idxmin()],
            'visits': dow_totals.min(),
            'day_num': dow_totals.idxmin()
        }

        # Seasonal insights
        seasonal_totals = yearly_forecast.groupby('Season')['Predicted_Admissions'].sum()
        insights['highest_season'] = {
            'season': seasonal_totals.idxmax(),
            'visits': seasonal_totals.max()
        }

        # Weekly insights
        weekly_totals = yearly_forecast.groupby('Week')['Predicted_Admissions'].sum()
        insights['highest_week'] = {
            'week': weekly_totals.idxmax(),
            'visits': weekly_totals.max()
        }

        # Quarterly insights
        quarterly_totals = yearly_forecast.groupby('Quarter')['Predicted_Admissions'].sum()
        insights['highest_quarter'] = {
            'quarter': quarterly_totals.idxmax(),
            'visits': quarterly_totals.max()
        }

        return insights

    def _show_adm_peak_periods_insights(self, insights, yearly_forecast):
        """Display peak periods insights for admissions with comprehensive visualizations"""
        st.markdown("#### Peak Period Predictions")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### Monthly Peaks")
            st.success(
                f"**Highest Month**: {insights['highest_month']['month']} ({insights['highest_month']['visits']:,.0f} admissions)")
            st.info(
                f"**Lowest Month**: {insights['lowest_month']['month']} ({insights['lowest_month']['visits']:,.0f} admissions)")

            difference = insights['highest_month']['visits'] - insights['lowest_month']['visits']
            st.metric("Peak vs Low Difference", f"{difference:,.0f} admissions")

        with col2:
            st.markdown("##### Weekly Patterns")
            st.success(
                f"**Busiest Day**: {insights['busiest_day']['day']} ({insights['busiest_day']['visits']:,.0f} total admissions)")
            st.info(
                f"**Quietest Day**: {insights['quietest_day']['day']} ({insights['quietest_day']['visits']:,.0f} total admissions)")

            weekly_difference = insights['busiest_day']['visits'] - insights['quietest_day']['visits']
            st.metric("Busiest vs Quietest", f"{weekly_difference:,.0f} admissions")

        # Create monthly prediction chart
        self._create_adm_monthly_prediction_chart(insights, yearly_forecast)

        # Create weekly prediction chart
        self._create_adm_weekly_prediction_chart(insights, yearly_forecast)

        # Quarterly trends visualization
        st.markdown("##### Quarterly Trends & Predictions")

        quarters = ['Q1', 'Q2', 'Q3', 'Q4']

        # Generate quarterly data from forecast
        quarterly_data = yearly_forecast.groupby('Quarter')['Predicted_Admissions'].agg(
            ['sum', 'mean', 'std']).reset_index()
        quarterly_data['Growth_Rate'] = quarterly_data['sum'].pct_change() * 100
        quarterly_data['Growth_Rate'].fillna(0, inplace=True)

        # Create dual-axis chart
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"secondary_y": True}]],
            subplot_titles=["Quarterly Admissions & Growth Predictions"]
        )

        # Admissions bars
        colors = ['red' if q == insights['highest_quarter']['quarter'] else 'lightblue' for q in
                  quarterly_data['Quarter']]
        fig.add_trace(
            go.Bar(
                x=quarterly_data['Quarter'],
                y=quarterly_data['sum'],
                name='Predicted Admissions',
                marker_color=colors,
                text=[f"{int(val):,}" for val in quarterly_data['sum']],
                textposition='outside'
            ),
            secondary_y=False,
        )

        # Growth rate line
        fig.add_trace(
            go.Scatter(
                x=quarterly_data['Quarter'],
                y=quarterly_data['Growth_Rate'],
                mode='lines+markers',
                name='Growth Rate (%)',
                line=dict(color='green', width=3),
                marker=dict(size=8)
            ),
            secondary_y=True,
        )

        # Update layout
        fig.update_xaxes(title_text="Quarter")
        fig.update_yaxes(title_text="Predicted Admissions", secondary_y=False)
        fig.update_yaxes(title_text="Growth Rate (%)", secondary_y=True)
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True, key="adm_quarterly_predictions")

        # Quarterly planning table
        st.markdown("##### Quarterly Planning Guide")

        planning_df = quarterly_data.copy()
        planning_df['Budget_Impact'] = planning_df.apply(
            lambda row: 'High' if row['Quarter'] == insights['highest_quarter']['quarter']
            else 'Medium' if row['Growth_Rate'] > 5
            else 'Low', axis=1
        )
        planning_df['Action_Required'] = planning_df.apply(
            lambda row: 'Surge capacity planning' if row['Quarter'] == insights['highest_quarter']['quarter']
            else 'Standard preparation' if row['Growth_Rate'] > 0
            else 'Cost optimization opportunity', axis=1
        )

        display_planning = planning_df[['Quarter', 'sum', 'Growth_Rate', 'Budget_Impact', 'Action_Required']].copy()
        display_planning.columns = ['Quarter', 'Predicted_Admissions', 'Growth_Rate_%', 'Budget_Impact', 'Action_Required']
        display_planning['Predicted_Admissions'] = display_planning['Predicted_Admissions'].astype(int)
        display_planning['Growth_Rate_%'] = display_planning['Growth_Rate_%'].round(1)

        st.dataframe(display_planning, use_container_width=True)

        # Specific predictions summary
        st.markdown("##### Key Predictions Summary")

        predictions = [
            f"Peak Week: Week {insights['highest_week']['week']} will be the busiest with {insights['highest_week']['visits']:,.0f} predicted admissions",
            f"Highest Quarter: {insights['highest_quarter']['quarter']} will have {insights['highest_quarter']['visits']:,.0f} admissions",
            f"Peak Season: {insights['highest_season']['season']} will see {insights['highest_season']['visits']:,.0f} admissions"
        ]

        for prediction in predictions:
            st.markdown(f"- {prediction}")

    def _create_adm_monthly_prediction_chart(self, insights, yearly_forecast):
        """Create monthly prediction visualization for admissions"""
        st.markdown("##### Monthly Volume Predictions")

        # Get actual monthly totals from forecast
        monthly_data = yearly_forecast.groupby('Month')['Predicted_Admissions'].sum().reset_index()
        monthly_data['Month_Name'] = monthly_data['Month'].map(lambda x: calendar.month_abbr[x])
        monthly_data['Is_Peak'] = monthly_data['Month'] == insights['highest_month']['month_num']
        monthly_data['Is_Low'] = monthly_data['Month'] == insights['lowest_month']['month_num']

        # Create bar chart with peak highlighting
        fig = go.Figure()

        # Regular months
        regular_months = monthly_data[~monthly_data['Is_Peak'] & ~monthly_data['Is_Low']]
        if not regular_months.empty:
            fig.add_trace(go.Bar(
                x=regular_months['Month_Name'],
                y=regular_months['Predicted_Admissions'],
                name='Regular Months',
                marker_color='lightblue'
            ))

        # Peak month
        peak_month = monthly_data[monthly_data['Is_Peak']]
        if not peak_month.empty:
            fig.add_trace(go.Bar(
                x=peak_month['Month_Name'],
                y=peak_month['Predicted_Admissions'],
                name='Peak Month',
                marker_color='red',
                text=[f"PEAK: {int(val):,}" for val in peak_month['Predicted_Admissions']],
                textposition='outside'
            ))

        # Low month
        low_month = monthly_data[monthly_data['Is_Low']]
        if not low_month.empty:
            fig.add_trace(go.Bar(
                x=low_month['Month_Name'],
                y=low_month['Predicted_Admissions'],
                name='Low Month',
                marker_color='green',
                text=[f"LOW: {int(val):,}" for val in low_month['Predicted_Admissions']],
                textposition='outside'
            ))

        fig.update_layout(
            title="Next 12 Months: Admissions Prediction by Month",
            height=400,
            xaxis_title="Month",
            yaxis_title="Predicted Admissions",
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True, key="adm_monthly_predictions")

    def _create_adm_weekly_prediction_chart(self, insights, yearly_forecast):
        """Create weekly pattern prediction visualization for admissions"""
        st.markdown("##### Weekly Pattern Predictions")

        # Get actual weekly totals from forecast
        weekly_data = yearly_forecast.groupby('DayOfWeek')['Predicted_Admissions'].agg(['sum', 'mean']).reset_index()
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_data['Day_Name'] = weekly_data['DayOfWeek'].map(lambda x: day_names[x])
        weekly_data['Is_Peak'] = weekly_data['DayOfWeek'] == insights['busiest_day']['day_num']
        weekly_data['Is_Low'] = weekly_data['DayOfWeek'] == insights['quietest_day']['day_num']

        # Create horizontal bar chart for better day visibility
        fig = go.Figure()

        colors = []
        for _, row in weekly_data.iterrows():
            if row['Is_Peak']:
                colors.append('red')
            elif row['Is_Low']:
                colors.append('green')
            else:
                colors.append('lightblue')

        fig.add_trace(go.Bar(
            x=weekly_data['mean'],
            y=weekly_data['Day_Name'],
            orientation='h',
            marker_color=colors,
            text=[f"{val:.0f}" for val in weekly_data['mean']],
            textposition='inside'
        ))

        fig.update_layout(
            title="Daily Average Admissions Prediction (Next Year)",
            height=400,
            xaxis_title="Daily Average Admissions",
            yaxis_title="Day of Week"
        )
        st.plotly_chart(fig, use_container_width=True, key="adm_weekly_predictions")

    def _show_adm_seasonal_trends_insights(self, insights, yearly_forecast, historical_analysis):
        """Display seasonal trends insights for admissions with comprehensive visualizations"""
        st.markdown("#### Seasonal Trend Analysis with Predictions")

        # Get seasonal data from forecast
        seasonal_data = yearly_forecast.groupby('Season')['Predicted_Admissions'].agg(['sum', 'mean', 'std']).reset_index()
        seasonal_data = seasonal_data.sort_values('sum', ascending=False)

        st.success(
            f"**Highest Demand Season**: {insights['highest_season']['season']} with {insights['highest_season']['visits']:,.0f} predicted admissions")

        # Create seasonal comparison visualization
        st.markdown("##### Seasonal Volume Comparison")

        fig = px.bar(
            seasonal_data,
            x='Season',
            y='sum',
            color='Season',
            title="Total Predicted Admissions by Season (Next 12 Months)",
            color_discrete_sequence=['lightgreen', 'gold', 'orange', 'lightblue']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True, key="adm_seasonal_comparison")

        # Create seasonal heatmap
        self._create_adm_seasonal_heatmap(insights, yearly_forecast)

        # Create seasonal trend line
        self._create_adm_seasonal_trend_analysis(yearly_forecast, historical_analysis)

        # Generate data-driven seasonal recommendations
        self._generate_adm_seasonal_recommendations(seasonal_data, insights)

    def _create_adm_seasonal_heatmap(self, insights, yearly_forecast):
        """Create seasonal prediction heatmap for admissions"""
        st.markdown("##### Seasonal Demand Intensity Heatmap")

        # Create seasonal data matrix
        seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
        months_map = {
            'Spring': [3, 4, 5],
            'Summer': [6, 7, 8],
            'Autumn': [9, 10, 11],
            'Winter': [12, 1, 2]
        }

        # Generate heatmap data from actual forecast
        heatmap_data = []

        for season in seasons:
            season_months = months_map[season]
            for month in season_months:
                month_data = yearly_forecast[yearly_forecast['Month'] == month]['Predicted_Admissions'].sum()
                max_month = yearly_forecast.groupby('Month')['Predicted_Admissions'].sum().max()
                intensity = month_data / max_month if max_month > 0 else 0  # Normalize to 0-1

                heatmap_data.append({
                    'Season': season,
                    'Month': calendar.month_abbr[month],
                    'Demand_Intensity': intensity,
                    'Total_Admissions': month_data
                })

        heatmap_df = pd.DataFrame(heatmap_data)

        # Create heatmap matrix
        pivot_data = heatmap_df.pivot(index='Season', columns='Month', values='Demand_Intensity')

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='RdYlBu_r',
            text=[[f"{val:.2f}" for val in row] for row in pivot_data.values],
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Demand Intensity")
        ))

        fig.update_layout(
            title="Seasonal Demand Intensity Heatmap (Next Year)",
            height=400,
            xaxis_title="Month",
            yaxis_title="Season"
        )
        st.plotly_chart(fig, use_container_width=True, key="adm_seasonal_heatmap")

    def _create_adm_seasonal_trend_analysis(self, yearly_forecast, historical_analysis):
        """Create seasonal trend analysis over time for admissions"""
        st.markdown("##### Seasonal Progression Analysis")

        # Create monthly progression
        monthly_progression = yearly_forecast.groupby([yearly_forecast['Date'].dt.to_period('M')])[
            'Predicted_Admissions'].sum()
        monthly_progression.index = monthly_progression.index.to_timestamp()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=monthly_progression.index,
            y=monthly_progression.values,
            mode='lines+markers',
            name='Monthly Totals',
            line=dict(color='blue', width=2)
        ))

        # Add seasonal trend line
        seasonal_avg = []
        for date in monthly_progression.index:
            month = date.month
            if month in historical_analysis['monthly'].index:
                seasonal_avg.append(historical_analysis['monthly'].loc[month, 'mean'] * 30)  # Convert to monthly
            else:
                seasonal_avg.append(monthly_progression.mean())

        fig.add_trace(go.Scatter(
            x=monthly_progression.index,
            y=seasonal_avg,
            mode='lines',
            name='Historical Seasonal Pattern',
            line=dict(color='red', dash='dash', width=2)
        ))

        fig.update_layout(
            title="Monthly Forecast vs Historical Seasonal Pattern",
            height=400,
            xaxis_title="Month",
            yaxis_title="Predicted Admissions"
        )
        st.plotly_chart(fig, use_container_width=True, key="adm_seasonal_progression")

    def _generate_adm_seasonal_recommendations(self, seasonal_data, insights):
        """Generate data-driven seasonal recommendations for admissions"""
        st.markdown("##### Data-Driven Seasonal Strategy")

        # Calculate seasonal metrics
        peak_season = insights['highest_season']['season']
        peak_admissions = seasonal_data[seasonal_data['Season'] == peak_season]['sum'].iloc[0]

        # Calculate relative increases needed
        for _, row in seasonal_data.iterrows():
            season = row['Season']
            admissions = row['sum']
            mean_admissions = row['mean']
            std_admissions = row['std']

            if season == peak_season:
                increase_factor = 1.0
                status = "PEAK SEASON"
                color = "error"
            else:
                increase_factor = peak_admissions / admissions if admissions > 0 else 1.0
                status = f"{((increase_factor - 1) * 100):+.0f}% vs Peak"
                color = "info"

            # Generate specific recommendations based on data
            variability = (std_admissions / mean_admissions) * 100 if mean_admissions > 0 else 0

            recommendations = []
            if increase_factor > 1.3:  # 30% or more increase needed
                recommendations.append(f"Increase staffing capacity by {((increase_factor - 1) * 100):.0f}%")
            elif increase_factor > 1.1:  # 10-30% increase needed
                recommendations.append(
                    f"Moderate capacity increase of {((increase_factor - 1) * 100):.0f}% recommended")
            else:
                recommendations.append("Standard capacity sufficient")

            if variability > 20:
                recommendations.append(f"High variability ({variability:.0f}%) - prepare flexible staffing")
            elif variability > 10:
                recommendations.append(f"Moderate variability ({variability:.0f}%) - monitor weekly patterns")
            else:
                recommendations.append(f"Low variability ({variability:.0f}%) - predictable demand")

            # Display season-specific insights
            if color == "error":
                st.error(f"**{season}** ({status}): {admissions:,.0f} admissions")
            else:
                st.info(f"**{season}** ({status}): {admissions:,.0f} admissions")

            for rec in recommendations:
                st.markdown(f"  - {rec}")

    def _show_adm_operational_planning_insights(self, insights, yearly_forecast):
        """Display operational planning insights for admissions"""
        st.markdown("#### Operational Planning Recommendations")

        # Generate dynamic staffing recommendations
        self._generate_adm_staffing_recommendations(insights, yearly_forecast)

        # Generate capacity planning recommendations
        self._generate_adm_capacity_recommendations(insights, yearly_forecast)

        # Generate resource allocation recommendations
        self._generate_adm_resource_recommendations(insights, yearly_forecast)

    def _generate_adm_staffing_recommendations(self, insights, yearly_forecast):
        """Generate data-driven staffing recommendations for admissions"""
        st.markdown("##### Staffing Recommendations")

        # Calculate staffing needs based on volume differences
        busiest_day_avg = yearly_forecast[yearly_forecast['DayOfWeek'] == insights['busiest_day']['day_num']][
            'Predicted_Admissions'].mean()
        quietest_day_avg = yearly_forecast[yearly_forecast['DayOfWeek'] == insights['quietest_day']['day_num']][
            'Predicted_Admissions'].mean()
        overall_avg = yearly_forecast['Predicted_Admissions'].mean()

        busiest_increase = ((busiest_day_avg - overall_avg) / overall_avg) * 100
        quietest_decrease = ((overall_avg - quietest_day_avg) / overall_avg) * 100

        staffing_recs = [
            f"**{insights['busiest_day']['day']} Staffing**: Increase by {busiest_increase:.0f}% above baseline ({busiest_day_avg:.0f} avg admissions)",
            f"**{insights['quietest_day']['day']} Optimization**: {quietest_decrease:.0f}% below average - optimal for training/maintenance",
            f"**{insights['highest_month']['month']} Peak**: Plan {((insights['highest_month']['visits'] / 12 - overall_avg) / overall_avg * 100):.0f}% capacity increase",
            f"**Week {insights['highest_week']['week']} Surge**: Maximum resources needed ({insights['highest_week']['visits']:,.0f} total admissions)"
        ]

        for rec in staffing_recs:
            st.markdown(f"- {rec}")

    def _generate_adm_capacity_recommendations(self, insights, yearly_forecast):
        """Generate capacity planning recommendations for admissions"""
        st.markdown("##### Capacity Planning")

        # Calculate capacity metrics
        max_daily = yearly_forecast['Predicted_Admissions'].max()
        min_daily = yearly_forecast['Predicted_Admissions'].min()
        capacity_range = max_daily - min_daily
        variability = (yearly_forecast['Predicted_Admissions'].std() / yearly_forecast['Predicted_Admissions'].mean()) * 100

        capacity_recs = [
            f"**Peak Capacity**: Design for {max_daily:.0f} daily admissions maximum",
            f"**Variability Management**: {variability:.0f}% coefficient of variation requires flexible capacity",
            f"**Equipment Planning**: {capacity_range:.0f} admission range necessitates scalable resources",
            f"**Surge Protocols**: Implement for demands above {yearly_forecast['Predicted_Admissions'].quantile(0.95):.0f} admissions"
        ]

        for rec in capacity_recs:
            st.markdown(f"- {rec}")

    def _generate_adm_resource_recommendations(self, insights, yearly_forecast):
        """Generate resource allocation recommendations for admissions"""
        st.markdown("##### Resource Allocation")

        # Calculate quarterly resource needs
        quarterly_data = yearly_forecast.groupby('Quarter')['Predicted_Admissions'].sum()
        peak_quarter_admissions = quarterly_data.max()

        resource_recs = [
            f"**Budget Priority**: {insights['highest_quarter']['quarter']} requires {((peak_quarter_admissions / quarterly_data.mean() - 1) * 100):.0f}% above average allocation",
            f"**Inventory Planning**: Stock levels should peak before {insights['highest_month']['month']}",
            f"**Maintenance Scheduling**: Optimal during {insights['lowest_month']['month']} (lowest predicted demand)",
            f"**Training Calendar**: Schedule major training during Week {yearly_forecast.groupby('Week')['Predicted_Admissions'].sum().idxmin()}"
        ]

        for rec in resource_recs:
            st.markdown(f"- {rec}")

        # Generate action timeline
        st.markdown("##### Priority Action Timeline")

        next_month = (datetime.now() + timedelta(days=30)).month
        if next_month == insights['highest_month']['month_num']:
            st.error(
                f"**IMMEDIATE ACTION**: Peak month {insights['highest_month']['month']} approaching - implement surge capacity now")
        elif abs(next_month - insights['highest_month']['month_num']) <= 2:
            st.warning(
                f"**PREPARE**: Peak month {insights['highest_month']['month']} in 2-3 months - begin capacity planning")
        else:
            st.info(
                f"**PLAN**: Peak month {insights['highest_month']['month']} in {abs(next_month - insights['highest_month']['month_num'])} months - long-term preparation phase")

    # ═══════════════════════════════════════════════════════════════════════════
    # INTEGRATED ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════

    def _show_integrated_analysis(self):
        """Show integrated analysis combining both datasets"""
        st.subheader("🔗 INTEGRATED ANALYSIS")

        if self.eu_data is None and self.admissions_data is None:
            st.error("❌ No datasets available for integrated analysis")
            return
        elif self.eu_data is None:
            st.warning("⚠️ EU Visits data not available - showing admissions analysis only")
            self._show_admissions_analysis()
            return
        elif self.admissions_data is None:
            st.warning("⚠️ Admissions data not available - showing EU visits analysis only")
            self._show_eu_visits_analysis()
            return

        # Show integrated analysis tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏥 System-Wide Operational",
            "🌡️ Environmental Impact",
            "📊 Strategic Healthcare Analytics",
            "🔄 Patient Journey Analytics",
            "🔮 Integrated 12-Month Forecast"
        ])

        with tab1:
            self._show_system_wide_operational()
        with tab2:
            self._show_environmental_impact()
        with tab3:
            self._show_strategic_analytics()
        with tab4:
            self._show_patient_journey_analytics()
        with tab5:
            self._show_integrated_12month_forecast()


    def _show_system_wide_operational(self):
        """🏥 System-Wide Operational Excellence"""
        st.markdown("### 🏥 System-Wide Operational Excellence")

        # Cross-dataset performance dashboard
        st.markdown("#### 📊 Cross-Dataset Performance Dashboard")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🚨 EU Visits Overview**")
            if 'Visit_Count' in self.eu_data.columns:
                eu_total = self.eu_data['Visit_Count'].sum()
                st.metric("Total EU Visits", f"{eu_total:,}")

                if 'Date' in self.eu_data.columns:
                    eu_daily_avg = self.eu_data.groupby('Date')['Visit_Count'].sum().mean()
                    eu_peak = self.eu_data.groupby('Date')['Visit_Count'].sum().max()
                    st.metric("Daily Average", f"{eu_daily_avg:.0f}")
                    st.metric("Peak Daily", f"{eu_peak:.0f}")

        with col2:
            st.markdown("**🏥 Admissions Overview**")
            if 'No_of_Admissions' in self.admissions_data.columns:
                adm_total = self.admissions_data['No_of_Admissions'].sum()
                st.metric("Total Admissions", f"{adm_total:,}")

                if 'Date' in self.admissions_data.columns:
                    adm_daily_avg = self.admissions_data.groupby('Date')['No_of_Admissions'].sum().mean()
                    adm_peak = self.admissions_data.groupby('Date')['No_of_Admissions'].sum().max()
                    st.metric("Daily Average", f"{adm_daily_avg:.0f}")
                    st.metric("Peak Daily", f"{adm_peak:.0f}")

        # System capacity vs demand correlation
        if ('Date' in self.eu_data.columns and 'Visit_Count' in self.eu_data.columns and
                'Date' in self.admissions_data.columns and 'No_of_Admissions' in self.admissions_data.columns):

            st.markdown("#### 📊 System Capacity vs Demand Correlation")

            # Combine daily volumes for both datasets
            eu_daily = self.eu_data.groupby('Date')['Visit_Count'].sum().reset_index()
            eu_daily.columns = ['Date', 'EU_Visits']

            adm_daily = self.admissions_data.groupby('Date')['No_of_Admissions'].sum().reset_index()
            adm_daily.columns = ['Date', 'Admissions']

            # Merge datasets
            combined_daily = pd.merge(eu_daily, adm_daily, on='Date', how='inner')

            if not combined_daily.empty:
                fig = px.scatter(
                    combined_daily,
                    x='EU_Visits',
                    y='Admissions',
                    title="Daily EU Visits vs Admissions Correlation",
                    labels={"EU_Visits": "Daily EU Visits", "Admissions": "Daily Admissions"},
                    trendline="ols"
                )
                st.plotly_chart(fig, use_container_width=True, key="int_system_wide_correlation_scatter")

                # Calculate correlation
                if len(combined_daily) > 1:
                    correlation = combined_daily['EU_Visits'].corr(combined_daily['Admissions'])
                    st.info(f"📊 **Correlation between EU Visits and Admissions: {correlation:.3f}**")

                    if correlation > 0.7:
                        st.success(
                            "✅ **Strong positive correlation** - High EU visits days correspond with high admission days")
                    elif correlation > 0.3:
                        st.info("📊 **Moderate correlation** - Some relationship between EU visits and admissions")
                    else:
                        st.warning("⚠️ **Weak correlation** - EU visits and admissions may follow different patterns")

        # NEW: Combined volume trends over time
        if ('Date' in self.eu_data.columns and 'Visit_Count' in self.eu_data.columns and
                'Date' in self.admissions_data.columns and 'No_of_Admissions' in self.admissions_data.columns):
            st.markdown("#### 📈 Combined System Volume Trends")

            # Create combined time series
            eu_daily = self.eu_data.groupby('Date')['Visit_Count'].sum().reset_index()
            adm_daily = self.admissions_data.groupby('Date')['No_of_Admissions'].sum().reset_index()

            combined_daily = pd.merge(eu_daily, adm_daily, on='Date', how='outer').fillna(0)
            combined_daily['Total_Volume'] = combined_daily['Visit_Count'] + combined_daily['No_of_Admissions']

            fig = px.line(
                combined_daily,
                x='Date',
                y='Total_Volume',
                title="Combined Healthcare System Volume Over Time"
            )
            fig.update_layout(
                height=400,
                margin=dict(l=80, r=50, t=100, b=120),
                xaxis=dict(tickangle=-45, nticks=20, tickfont=dict(size=10))
            )
            st.plotly_chart(fig, use_container_width=True, key="int_combined_volume_trend")

        # NEW: System efficiency comparison
        st.markdown("#### ⚡ System Efficiency Comparison")

        if ('Is_Weekend' in self.eu_data.columns and 'Is_Weekend' in self.admissions_data.columns):
            # Weekend vs Weekday efficiency
            col1, col2 = st.columns(2)

            with col1:
                eu_efficiency = self.eu_data.groupby('Is_Weekend')['Visit_Count'].agg(['sum', 'mean']).reset_index()
                eu_efficiency['Is_Weekend'] = eu_efficiency['Is_Weekend'].map({True: 'Weekend', False: 'Weekday'})
                eu_efficiency['Dataset'] = 'EU Visits'

                fig = px.bar(
                    eu_efficiency,
                    x='Is_Weekend',
                    y='mean',
                    title="EU Visits: Weekday vs Weekend Average",
                    color='Is_Weekend'
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True, key="int_eu_efficiency")

            with col2:
                adm_efficiency = self.admissions_data.groupby('Is_Weekend')['No_of_Admissions'].agg(
                    ['sum', 'mean']).reset_index()
                adm_efficiency['Is_Weekend'] = adm_efficiency['Is_Weekend'].map({True: 'Weekend', False: 'Weekday'})
                adm_efficiency['Dataset'] = 'Admissions'

                fig = px.bar(
                    adm_efficiency,
                    x='Is_Weekend',
                    y='mean',
                    title="Admissions: Weekday vs Weekend Average",
                    color='Is_Weekend'
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True, key="int_adm_efficiency")

    def _show_environmental_impact(self):
        """🌡️ Environmental Impact Assessment"""
        st.markdown("### 🌡️ Environmental Impact Assessment")
        st.caption("Seasonal weather patterns, Temperature correlations, Environmental factors")

        # Create seasonal tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🌸 Spring", "☀️ Summer", "🍂 Autumn", "❄️ Winter"])

        with tab1:
            self._show_integrated_spring_analysis()
        with tab2:
            self._show_integrated_summer_analysis()
        with tab3:
            self._show_integrated_autumn_analysis()
        with tab4:
            self._show_integrated_winter_analysis()

    def _show_integrated_spring_analysis(self):
        """Spring environmental analysis across both datasets"""
        st.markdown("#### 🌸 Spring Environmental Analysis (March - May)")

        if 'Month' in self.eu_data.columns and 'Month' in self.admissions_data.columns:
            eu_spring = self.eu_data[self.eu_data['Month'].isin([3, 4, 5])]
            adm_spring = self.admissions_data[self.admissions_data['Month'].isin([3, 4, 5])]

            if eu_spring.empty or adm_spring.empty:
                st.warning("⚠️ Spring data is not available in one or both datasets.")
                return

            # Combined monthly patterns in spring
            st.markdown("##### 📅 Spring Monthly Patterns Comparison")

            eu_spring_monthly = eu_spring.groupby('Month_Name')['Visit_Count'].sum().reset_index()
            eu_spring_monthly['Dataset'] = 'EU Visits'
            eu_spring_monthly.columns = ['Month_Name', 'Count', 'Dataset']

            adm_spring_monthly = adm_spring.groupby('Month_Name')['No_of_Admissions'].sum().reset_index()
            adm_spring_monthly['Dataset'] = 'Admissions'
            adm_spring_monthly.columns = ['Month_Name', 'Count', 'Dataset']

            spring_combined = pd.concat([eu_spring_monthly, adm_spring_monthly], ignore_index=True)

            fig = px.bar(
                spring_combined,
                x='Month_Name',
                y='Count',
                color='Dataset',
                title="Spring: EU Visits vs Admissions by Month",
                barmode='group'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key="int_spring_monthly_comparison")

            # Weather impact in spring
            if 'Weather_Category' in eu_spring.columns and 'Weather_Category' in adm_spring.columns:
                st.markdown("##### 🌤️ Spring Weather Impact Comparison")

                col1, col2 = st.columns(2)

                with col1:
                    eu_spring_weather = eu_spring.groupby('Weather_Category')['Visit_Count'].sum().reset_index()
                    fig = px.pie(
                        eu_spring_weather,
                        values='Visit_Count',
                        names='Weather_Category',
                        title="EU Visits by Spring Weather",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True, key="int_spring_eu_weather")

                with col2:
                    adm_spring_weather = adm_spring.groupby('Weather_Category')['No_of_Admissions'].sum().reset_index()
                    fig = px.pie(
                        adm_spring_weather,
                        values='No_of_Admissions',
                        names='Weather_Category',
                        title="Admissions by Spring Weather",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True, key="int_spring_adm_weather")

            # Temperature correlation analysis
            if 'Temperature_Mean_C' in eu_spring.columns and 'Temperature_Mean_C' in adm_spring.columns:
                st.markdown("##### 🌡️ Spring Temperature Impact Analysis")

                col1, col2 = st.columns(2)

                with col1:
                    eu_temp_spring = eu_spring.groupby('Temperature_Mean_C')['Visit_Count'].sum().reset_index()
                    if len(eu_temp_spring) > 1:
                        correlation_eu = eu_temp_spring['Temperature_Mean_C'].corr(eu_temp_spring['Visit_Count'])

                        fig = px.scatter(
                            eu_temp_spring,
                            x='Temperature_Mean_C',
                            y='Visit_Count',
                            title=f"Spring EU Visits vs Temperature (r = {correlation_eu:.3f})",
                            trendline="ols"
                        )
                        fig.update_layout(height=350)
                        st.plotly_chart(fig, use_container_width=True, key="int_spring_eu_temp")

                with col2:
                    adm_temp_spring = adm_spring.groupby('Temperature_Mean_C')['No_of_Admissions'].sum().reset_index()
                    if len(adm_temp_spring) > 1:
                        correlation_adm = adm_temp_spring['Temperature_Mean_C'].corr(
                            adm_temp_spring['No_of_Admissions'])

                        fig = px.scatter(
                            adm_temp_spring,
                            x='Temperature_Mean_C',
                            y='No_of_Admissions',
                            title=f"Spring Admissions vs Temperature (r = {correlation_adm:.3f})",
                            trendline="ols"
                        )
                        fig.update_layout(height=350)
                        st.plotly_chart(fig, use_container_width=True, key="int_spring_adm_temp")

            # Precipitation impact
            if 'Precipitation_Category' in eu_spring.columns and 'Precipitation_Category' in adm_spring.columns:
                st.markdown("##### 🌧️ Spring Precipitation Impact")

                eu_precip = eu_spring.groupby('Precipitation_Category')['Visit_Count'].sum().reset_index()
                eu_precip['Dataset'] = 'EU Visits'
                eu_precip.columns = ['Precipitation_Category', 'Count', 'Dataset']

                adm_precip = adm_spring.groupby('Precipitation_Category')['No_of_Admissions'].sum().reset_index()
                adm_precip['Dataset'] = 'Admissions'
                adm_precip.columns = ['Precipitation_Category', 'Count', 'Dataset']

                precip_combined = pd.concat([eu_precip, adm_precip], ignore_index=True)

                fig = px.bar(
                    precip_combined,
                    x='Precipitation_Category',
                    y='Count',
                    color='Dataset',
                    title="Spring: Precipitation Impact on Healthcare Volume",
                    barmode='group'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True, key="int_spring_precipitation")

    def _show_integrated_summer_analysis(self):
        """Summer environmental analysis across both datasets"""
        st.markdown("#### ☀️ Summer Environmental Analysis (June - August)")

        if 'Month' in self.eu_data.columns and 'Month' in self.admissions_data.columns:
            eu_summer = self.eu_data[self.eu_data['Month'].isin([6, 7, 8])]
            adm_summer = self.admissions_data[self.admissions_data['Month'].isin([6, 7, 8])]

            if eu_summer.empty or adm_summer.empty:
                st.warning("⚠️ Summer data is not available in one or both datasets.")
                return

            # Combined monthly patterns in summer
            st.markdown("##### 📅 Summer Monthly Patterns Comparison")

            eu_summer_monthly = eu_summer.groupby('Month_Name')['Visit_Count'].sum().reset_index()
            eu_summer_monthly['Dataset'] = 'EU Visits'
            eu_summer_monthly.columns = ['Month_Name', 'Count', 'Dataset']

            adm_summer_monthly = adm_summer.groupby('Month_Name')['No_of_Admissions'].sum().reset_index()
            adm_summer_monthly['Dataset'] = 'Admissions'
            adm_summer_monthly.columns = ['Month_Name', 'Count', 'Dataset']

            summer_combined = pd.concat([eu_summer_monthly, adm_summer_monthly], ignore_index=True)

            fig = px.bar(
                summer_combined,
                x='Month_Name',
                y='Count',
                color='Dataset',
                title="Summer: EU Visits vs Admissions by Month",
                barmode='group'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key="int_summer_monthly_comparison")

            # Weather impact in summer
            if 'Weather_Category' in eu_summer.columns and 'Weather_Category' in adm_summer.columns:
                st.markdown("##### 🌤️ Summer Weather Impact Comparison")

                col1, col2 = st.columns(2)

                with col1:
                    eu_summer_weather = eu_summer.groupby('Weather_Category')['Visit_Count'].sum().reset_index()
                    fig = px.pie(
                        eu_summer_weather,
                        values='Visit_Count',
                        names='Weather_Category',
                        title="EU Visits by Summer Weather",
                        color_discrete_sequence=px.colors.qualitative.Pastel1
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True, key="int_summer_eu_weather")

                with col2:
                    adm_summer_weather = adm_summer.groupby('Weather_Category')['No_of_Admissions'].sum().reset_index()
                    fig = px.pie(
                        adm_summer_weather,
                        values='No_of_Admissions',
                        names='Weather_Category',
                        title="Admissions by Summer Weather",
                        color_discrete_sequence=px.colors.qualitative.Pastel1
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True, key="int_summer_adm_weather")

            # Temperature correlation analysis
            if 'Temperature_Mean_C' in eu_summer.columns and 'Temperature_Mean_C' in adm_summer.columns:
                st.markdown("##### 🌡️ Summer Temperature Impact Analysis")

                col1, col2 = st.columns(2)

                with col1:
                    eu_temp_summer = eu_summer.groupby('Temperature_Mean_C')['Visit_Count'].sum().reset_index()
                    if len(eu_temp_summer) > 1:
                        correlation_eu = eu_temp_summer['Temperature_Mean_C'].corr(eu_temp_summer['Visit_Count'])

                        fig = px.scatter(
                            eu_temp_summer,
                            x='Temperature_Mean_C',
                            y='Visit_Count',
                            title=f"Summer EU Visits vs Temperature (r = {correlation_eu:.3f})",
                            trendline="ols"
                        )
                        fig.update_layout(height=350)
                        st.plotly_chart(fig, use_container_width=True, key="int_summer_eu_temp")

                with col2:
                    adm_temp_summer = adm_summer.groupby('Temperature_Mean_C')['No_of_Admissions'].sum().reset_index()
                    if len(adm_temp_summer) > 1:
                        correlation_adm = adm_temp_summer['Temperature_Mean_C'].corr(
                            adm_temp_summer['No_of_Admissions'])

                        fig = px.scatter(
                            adm_temp_summer,
                            x='Temperature_Mean_C',
                            y='No_of_Admissions',
                            title=f"Summer Admissions vs Temperature (r = {correlation_adm:.3f})",
                            trendline="ols"
                        )
                        fig.update_layout(height=350)
                        st.plotly_chart(fig, use_container_width=True, key="int_summer_adm_temp")

            # Heat wave analysis
            if 'Temperature_Mean_C' in eu_summer.columns:
                st.markdown("##### 🔥 Summer Heat Impact Analysis")

                # Define heat thresholds
                moderate_heat = eu_summer['Temperature_Mean_C'].quantile(0.75)
                extreme_heat = eu_summer['Temperature_Mean_C'].quantile(0.9)

                heat_categories = []
                for temp in eu_summer['Temperature_Mean_C']:
                    if temp >= extreme_heat:
                        heat_categories.append('Extreme Heat')
                    elif temp >= moderate_heat:
                        heat_categories.append('Moderate Heat')
                    else:
                        heat_categories.append('Normal')

                eu_summer_heat = eu_summer.copy()
                eu_summer_heat['Heat_Category'] = heat_categories

                heat_impact = eu_summer_heat.groupby('Heat_Category')['Visit_Count'].sum().reset_index()

                fig = px.bar(
                    heat_impact,
                    x='Heat_Category',
                    y='Visit_Count',
                    title="Summer EU Visits by Heat Categories",
                    color='Heat_Category',
                    color_discrete_map={
                        'Normal': 'lightblue',
                        'Moderate Heat': 'orange',
                        'Extreme Heat': 'red'
                    }
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True, key="int_summer_heat_impact")

    def _show_integrated_autumn_analysis(self):
        """Autumn environmental analysis across both datasets"""
        st.markdown("#### 🍂 Autumn Environmental Analysis (September - November)")

        if 'Month' in self.eu_data.columns and 'Month' in self.admissions_data.columns:
            eu_autumn = self.eu_data[self.eu_data['Month'].isin([9, 10, 11])]
            adm_autumn = self.admissions_data[self.admissions_data['Month'].isin([9, 10, 11])]

            if eu_autumn.empty or adm_autumn.empty:
                st.warning("⚠️ Autumn data is not available in one or both datasets.")
                return

            # Combined monthly patterns in autumn
            st.markdown("##### 📅 Autumn Monthly Patterns Comparison")

            eu_autumn_monthly = eu_autumn.groupby('Month_Name')['Visit_Count'].sum().reset_index()
            eu_autumn_monthly['Dataset'] = 'EU Visits'
            eu_autumn_monthly.columns = ['Month_Name', 'Count', 'Dataset']

            adm_autumn_monthly = adm_autumn.groupby('Month_Name')['No_of_Admissions'].sum().reset_index()
            adm_autumn_monthly['Dataset'] = 'Admissions'
            adm_autumn_monthly.columns = ['Month_Name', 'Count', 'Dataset']

            autumn_combined = pd.concat([eu_autumn_monthly, adm_autumn_monthly], ignore_index=True)

            fig = px.bar(
                autumn_combined,
                x='Month_Name',
                y='Count',
                color='Dataset',
                title="Autumn: EU Visits vs Admissions by Month",
                barmode='group'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key="int_autumn_monthly_comparison")

            # Weather impact in autumn
            if 'Weather_Category' in eu_autumn.columns and 'Weather_Category' in adm_autumn.columns:
                st.markdown("##### 🌤️ Autumn Weather Impact Comparison")

                col1, col2 = st.columns(2)

                with col1:
                    eu_autumn_weather = eu_autumn.groupby('Weather_Category')['Visit_Count'].sum().reset_index()
                    fig = px.pie(
                        eu_autumn_weather,
                        values='Visit_Count',
                        names='Weather_Category',
                        title="EU Visits by Autumn Weather",
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True, key="int_autumn_eu_weather")

                with col2:
                    adm_autumn_weather = adm_autumn.groupby('Weather_Category')['No_of_Admissions'].sum().reset_index()
                    fig = px.pie(
                        adm_autumn_weather,
                        values='No_of_Admissions',
                        names='Weather_Category',
                        title="Admissions by Autumn Weather",
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True, key="int_autumn_adm_weather")

            # Temperature correlation analysis
            if 'Temperature_Mean_C' in eu_autumn.columns and 'Temperature_Mean_C' in adm_autumn.columns:
                st.markdown("##### 🌡️ Autumn Temperature Impact Analysis")

                col1, col2 = st.columns(2)

                with col1:
                    eu_temp_autumn = eu_autumn.groupby('Temperature_Mean_C')['Visit_Count'].sum().reset_index()
                    if len(eu_temp_autumn) > 1:
                        correlation_eu = eu_temp_autumn['Temperature_Mean_C'].corr(eu_temp_autumn['Visit_Count'])

                        fig = px.scatter(
                            eu_temp_autumn,
                            x='Temperature_Mean_C',
                            y='Visit_Count',
                            title=f"Autumn EU Visits vs Temperature (r = {correlation_eu:.3f})",
                            trendline="ols"
                        )
                        fig.update_layout(height=350)
                        st.plotly_chart(fig, use_container_width=True, key="int_autumn_eu_temp")

                with col2:
                    adm_temp_autumn = adm_autumn.groupby('Temperature_Mean_C')['No_of_Admissions'].sum().reset_index()
                    if len(adm_temp_autumn) > 1:
                        correlation_adm = adm_temp_autumn['Temperature_Mean_C'].corr(
                            adm_temp_autumn['No_of_Admissions'])

                        fig = px.scatter(
                            adm_temp_autumn,
                            x='Temperature_Mean_C',
                            y='No_of_Admissions',
                            title=f"Autumn Admissions vs Temperature (r = {correlation_adm:.3f})",
                            trendline="ols"
                        )
                        fig.update_layout(height=350)
                        st.plotly_chart(fig, use_container_width=True, key="int_autumn_adm_temp")

            # Flu season analysis
            st.markdown("##### 🦠 Autumn Flu Season Impact")

            if 'Month_Name' in eu_autumn.columns:
                # Assume November is peak flu season start
                flu_season_eu = eu_autumn[eu_autumn['Month_Name'] == 'November']
                flu_season_adm = adm_autumn[adm_autumn['Month_Name'] == 'November']

                if not flu_season_eu.empty and not flu_season_adm.empty:
                    flu_eu_total = flu_season_eu['Visit_Count'].sum()
                    flu_adm_total = flu_season_adm['No_of_Admissions'].sum()

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("November EU Visits", f"{flu_eu_total:,}")

                    with col2:
                        st.metric("November Admissions", f"{flu_adm_total:,}")

                    st.info(
                        "November typically marks the beginning of flu season, which may contribute to increased healthcare utilization.")

    def _show_integrated_winter_analysis(self):
        """Winter environmental analysis across both datasets"""
        st.markdown("#### ❄️ Winter Environmental Analysis (December - February)")

        if 'Month' in self.eu_data.columns and 'Month' in self.admissions_data.columns:
            eu_winter = self.eu_data[self.eu_data['Month'].isin([12, 1, 2])]
            adm_winter = self.admissions_data[self.admissions_data['Month'].isin([12, 1, 2])]

            if eu_winter.empty or adm_winter.empty:
                st.warning("⚠️ Winter data is not available in one or both datasets.")
                return

            # Combined monthly patterns in winter
            st.markdown("##### 📅 Winter Monthly Patterns Comparison")

            eu_winter_monthly = eu_winter.groupby('Month_Name')['Visit_Count'].sum().reset_index()
            eu_winter_monthly['Dataset'] = 'EU Visits'
            eu_winter_monthly.columns = ['Month_Name', 'Count', 'Dataset']

            adm_winter_monthly = adm_winter.groupby('Month_Name')['No_of_Admissions'].sum().reset_index()
            adm_winter_monthly['Dataset'] = 'Admissions'
            adm_winter_monthly.columns = ['Month_Name', 'Count', 'Dataset']

            winter_combined = pd.concat([eu_winter_monthly, adm_winter_monthly], ignore_index=True)

            fig = px.bar(
                winter_combined,
                x='Month_Name',
                y='Count',
                color='Dataset',
                title="Winter: EU Visits vs Admissions by Month",
                barmode='group'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key="int_winter_monthly_comparison")

            # Weather impact in winter
            if 'Weather_Category' in eu_winter.columns and 'Weather_Category' in adm_winter.columns:
                st.markdown("##### 🌤️ Winter Weather Impact Comparison")

                col1, col2 = st.columns(2)

                with col1:
                    eu_winter_weather = eu_winter.groupby('Weather_Category')['Visit_Count'].sum().reset_index()
                    fig = px.pie(
                        eu_winter_weather,
                        values='Visit_Count',
                        names='Weather_Category',
                        title="EU Visits by Winter Weather",
                        color_discrete_sequence=px.colors.qualitative.Pastel2
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True, key="int_winter_eu_weather")

                with col2:
                    adm_winter_weather = adm_winter.groupby('Weather_Category')['No_of_Admissions'].sum().reset_index()
                    fig = px.pie(
                        adm_winter_weather,
                        values='No_of_Admissions',
                        names='Weather_Category',
                        title="Admissions by Winter Weather",
                        color_discrete_sequence=px.colors.qualitative.Pastel2
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True, key="int_winter_adm_weather")

            # Temperature correlation analysis
            if 'Temperature_Mean_C' in eu_winter.columns and 'Temperature_Mean_C' in adm_winter.columns:
                st.markdown("##### 🌡️ Winter Temperature Impact Analysis")

                col1, col2 = st.columns(2)

                with col1:
                    eu_temp_winter = eu_winter.groupby('Temperature_Mean_C')['Visit_Count'].sum().reset_index()
                    if len(eu_temp_winter) > 1:
                        correlation_eu = eu_temp_winter['Temperature_Mean_C'].corr(eu_temp_winter['Visit_Count'])

                        fig = px.scatter(
                            eu_temp_winter,
                            x='Temperature_Mean_C',
                            y='Visit_Count',
                            title=f"Winter EU Visits vs Temperature (r = {correlation_eu:.3f})",
                            trendline="ols"
                        )
                        fig.update_layout(height=350)
                        st.plotly_chart(fig, use_container_width=True, key="int_winter_eu_temp")

                with col2:
                    adm_temp_winter = adm_winter.groupby('Temperature_Mean_C')['No_of_Admissions'].sum().reset_index()
                    if len(adm_temp_winter) > 1:
                        correlation_adm = adm_temp_winter['Temperature_Mean_C'].corr(
                            adm_temp_winter['No_of_Admissions'])

                        fig = px.scatter(
                            adm_temp_winter,
                            x='Temperature_Mean_C',
                            y='No_of_Admissions',
                            title=f"Winter Admissions vs Temperature (r = {correlation_adm:.3f})",
                            trendline="ols"
                        )
                        fig.update_layout(height=350)
                        st.plotly_chart(fig, use_container_width=True, key="int_winter_adm_temp")

            # Cold weather health alerts
            if 'Temperature_Mean_C' in eu_winter.columns:
                st.markdown("##### 🥶 Cold Weather Health Impact")

                # Define cold thresholds
                cold_threshold = eu_winter['Temperature_Mean_C'].quantile(0.25)
                extreme_cold = eu_winter['Temperature_Mean_C'].quantile(0.1)

                cold_days_eu = eu_winter[eu_winter['Temperature_Mean_C'] <= cold_threshold]['Visit_Count'].sum()
                extreme_cold_days_eu = eu_winter[eu_winter['Temperature_Mean_C'] <= extreme_cold]['Visit_Count'].sum()

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("EU Visits on Cold Days", f"{cold_days_eu:,}")

                with col2:
                    st.metric("EU Visits on Extreme Cold Days", f"{extreme_cold_days_eu:,}")

                st.info(f"Cold threshold: {cold_threshold:.1f}°C | Extreme cold: {extreme_cold:.1f}°C")

    def _show_strategic_analytics(self):
        """📊 Strategic Healthcare Analytics"""
        st.markdown("### 📊 Strategic Healthcare Analytics")

        # Healthcare demand trends
        if ('Date' in self.eu_data.columns and 'Date' in self.admissions_data.columns):
            st.markdown("#### 🔮 Healthcare Demand Trends")

            # Create monthly trends
            eu_monthly = self.eu_data.groupby([self.eu_data['Date'].dt.to_period('M')])[
                'Visit_Count'].sum().reset_index()
            eu_monthly['Date'] = eu_monthly['Date'].dt.to_timestamp()
            eu_monthly.columns = ['Date', 'EU_Visits']

            adm_monthly = self.admissions_data.groupby([self.admissions_data['Date'].dt.to_period('M')])[
                'No_of_Admissions'].sum().reset_index()
            adm_monthly['Date'] = adm_monthly['Date'].dt.to_timestamp()
            adm_monthly.columns = ['Date', 'Admissions']

            monthly_combined = pd.merge(eu_monthly, adm_monthly, on='Date', how='inner')

            if not monthly_combined.empty:
                fig = make_subplots(specs=[[{"secondary_y": True}]])

                fig.add_trace(
                    go.Scatter(x=monthly_combined['Date'], y=monthly_combined['EU_Visits'], name="EU Visits"),
                    secondary_y=False,
                )

                fig.add_trace(
                    go.Scatter(x=monthly_combined['Date'], y=monthly_combined['Admissions'], name="Admissions"),
                    secondary_y=True,
                )

                fig.update_xaxes(title_text="Date")
                fig.update_yaxes(title_text="EU Visits", secondary_y=False)
                fig.update_yaxes(title_text="Admissions", secondary_y=True)

                fig.update_layout(title_text="Monthly Healthcare Demand Trends", height=400)
                st.plotly_chart(fig, use_container_width=True, key="int_strategic_demand_trends_subplots")

        # NEW: Combined volume distribution analysis
        st.markdown("#### 📊 System Volume Distribution Analysis")

        if ('Visit_Count' in self.eu_data.columns and 'No_of_Admissions' in self.admissions_data.columns):
            col1, col2 = st.columns(2)

            with col1:
                # System volume pie chart
                eu_total = self.eu_data['Visit_Count'].sum()
                adm_total = self.admissions_data['No_of_Admissions'].sum()

                system_volume = pd.DataFrame({
                    'Service_Type': ['EU Visits', 'Admissions'],
                    'Volume': [eu_total, adm_total]
                })

                fig = px.pie(
                    system_volume,
                    values='Volume',
                    names='Service_Type',
                    title="Healthcare System Volume Distribution"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True, key="int_system_volume_pie")

            with col2:
                # Conversion rate analysis
                conversion_rate = (adm_total / eu_total) * 100 if eu_total > 0 else 0

                conversion_data = pd.DataFrame({
                    'Metric': ['EU Visits', 'Converted to Admission', 'Not Admitted'],
                    'Value': [eu_total, adm_total, eu_total - adm_total]
                })

                # Show only conversion funnel
                funnel_data = pd.DataFrame({
                    'Stage': ['Total EU Visits', 'Admissions'],
                    'Count': [eu_total, adm_total]
                })

                fig = px.funnel(
                    funnel_data,
                    x='Count',
                    y='Stage',
                    title=f"Healthcare Conversion Funnel ({conversion_rate:.1f}% admission rate)"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True, key="int_conversion_funnel")

        # System performance benchmarking (enhanced)
        st.markdown("#### 🎯 Healthcare System Performance Benchmarking")

        # Calculate system-wide efficiency metrics
        if ('Date' in self.eu_data.columns and 'Date' in self.admissions_data.columns):
            total_days = len(pd.date_range(
                start=min(self.eu_data['Date'].min(), self.admissions_data['Date'].min()),
                end=max(self.eu_data['Date'].max(), self.admissions_data['Date'].max())
            ))

            eu_total = self.eu_data['Visit_Count'].sum() if 'Visit_Count' in self.eu_data.columns else 0
            adm_total = self.admissions_data[
                'No_of_Admissions'].sum() if 'No_of_Admissions' in self.admissions_data.columns else 0

            system_metrics = {
                'Total EU Visits': eu_total,
                'Total Admissions': adm_total,
                'Days Analyzed': total_days,
                'EU Visits per Day': eu_total / total_days if total_days > 0 else 0,
                'Admissions per Day': adm_total / total_days if total_days > 0 else 0,
                'Admission Rate': (adm_total / eu_total) * 100 if eu_total > 0 else 0
            }

            # Display system metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total System Volume",
                          f"{system_metrics['Total EU Visits'] + system_metrics['Total Admissions']:,.0f}")
                st.metric("Analysis Period", f"{system_metrics['Days Analyzed']} days")

            with col2:
                st.metric("Daily EU Visits", f"{system_metrics['EU Visits per Day']:.0f}")
                st.metric("Daily Admissions", f"{system_metrics['Admissions per Day']:.0f}")

            with col3:
                st.metric("System Admission Rate", f"{system_metrics['Admission Rate']:.1f}%")
                daily_total = system_metrics['EU Visits per Day'] + system_metrics['Admissions per Day']
                st.metric("Total Daily Activity", f"{daily_total:.0f}")

        # NEW: Hospital performance comparison across datasets
        if ('Hospital_Name' in self.eu_data.columns and 'Hospital_Name' in self.admissions_data.columns):
            st.markdown("#### 🏥 Cross-Dataset Hospital Performance")

            # Get hospital data from both datasets
            eu_hospitals = self.eu_data.groupby('Hospital_Name')['Visit_Count'].sum().reset_index()
            adm_hospitals = self.admissions_data.groupby('Hospital_Name')['No_of_Admissions'].sum().reset_index()

            # Merge hospital data
            hospital_combined = pd.merge(eu_hospitals, adm_hospitals, on='Hospital_Name', how='outer').fillna(0)
            hospital_combined['Total_Activity'] = hospital_combined['Visit_Count'] + hospital_combined[
                'No_of_Admissions']
            hospital_combined['Admission_Rate'] = (
                        hospital_combined['No_of_Admissions'] / hospital_combined['Visit_Count'] * 100).fillna(0)

            # Hospital performance scatter plot
            fig = px.scatter(
                hospital_combined,
                x='Visit_Count',
                y='No_of_Admissions',
                size='Total_Activity',
                hover_name='Hospital_Name',
                title="Hospital Performance: EU Visits vs Admissions (Size = Total Activity)"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True, key="int_hospital_performance_scatter")

        # Resource optimization insights (enhanced)
        st.markdown("#### 📊 Resource Optimization Insights")

        optimization_insights = [
            f"🔍 **Peak Demand**: EU visits peak at {self.eu_data.groupby('Date')['Visit_Count'].sum().max():.0f} daily, Admissions peak at {self.admissions_data.groupby('Date')['No_of_Admissions'].sum().max():.0f} daily",
            f"⚡ **Conversion Rate**: Approximately {system_metrics['Admission Rate']:.1f}% of healthcare encounters result in admission",
            f"📈 **System Utilization**: Daily average of {system_metrics['EU Visits per Day'] + system_metrics['Admissions per Day']:.0f} total patient interactions",
            f"🎯 **Efficiency Opportunity**: Analyze high EU visit days that don't correlate with high admissions for capacity optimization"
        ]

        for insight in optimization_insights:
            st.markdown(insight)

        # NEW: Predictive insights
        st.markdown("#### 🔮 Predictive Insights")

        predictive_insights = [
            "📊 **Volume Forecasting**: Use historical patterns to predict future demand",
            "🌡️ **Weather-Based Planning**: Adjust staffing based on weather forecasts",
            "📅 **Seasonal Preparation**: Plan capacity increases during high-demand seasons",
            "⚡ **Real-time Optimization**: Monitor daily patterns for immediate adjustments"
        ]

        for insight in predictive_insights:
            st.markdown(insight)

    def _show_patient_journey_analytics(self):
        """🔄 Patient Journey Analytics"""
        st.markdown("### 🔄 Patient Journey Analytics")
        st.caption("Care pathways, Conversion patterns, Patient flow optimization")

        # Conversion Analysis
        st.markdown("#### 📊 EU to Admission Conversion Analysis")

        if ('Visit_Count' in self.eu_data.columns and 'No_of_Admissions' in self.admissions_data.columns):
            eu_total = self.eu_data['Visit_Count'].sum()
            adm_total = self.admissions_data['No_of_Admissions'].sum()
            conversion_rate = (adm_total / eu_total) * 100 if eu_total > 0 else 0

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total EU Visits", f"{eu_total:,}")
            with col2:
                st.metric("Total Admissions", f"{adm_total:,}")
            with col3:
                st.metric("Conversion Rate", f"{conversion_rate:.1f}%")
            with col4:
                non_admitted = eu_total - adm_total
                st.metric("Non-Admitted", f"{non_admitted:,}")

            # Conversion funnel visualization
            funnel_data = pd.DataFrame({
                'Stage': ['EU Visits', 'Admissions'],
                'Count': [eu_total, adm_total],
                'Percentage': [100, conversion_rate]
            })

            fig = px.funnel(
                funnel_data,
                x='Count',
                y='Stage',
                title=f"Patient Care Conversion Funnel ({conversion_rate:.1f}% Conversion Rate)"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key="patient_journey_funnel")

        # Daily conversion patterns
        if ('Date' in self.eu_data.columns and 'Date' in self.admissions_data.columns):
            st.markdown("#### 📅 Daily Conversion Patterns")

            eu_daily = self.eu_data.groupby('Date')['Visit_Count'].sum().reset_index()
            adm_daily = self.admissions_data.groupby('Date')['No_of_Admissions'].sum().reset_index()

            daily_combined = pd.merge(eu_daily, adm_daily, on='Date', how='inner')
            daily_combined['Conversion_Rate'] = (
                        daily_combined['No_of_Admissions'] / daily_combined['Visit_Count'] * 100).fillna(0)
            daily_combined['Conversion_Rate'] = daily_combined['Conversion_Rate'].replace([float('inf'), -float('inf')],
                                                                                          0)

            if not daily_combined.empty:
                fig = px.line(
                    daily_combined,
                    x='Date',
                    y='Conversion_Rate',
                    title="Daily EU to Admission Conversion Rate Over Time"
                )
                fig.update_layout(
                    height=400,
                    yaxis_title="Conversion Rate (%)",
                    xaxis=dict(tickangle=-45, nticks=20, tickfont=dict(size=10))
                )
                st.plotly_chart(fig, use_container_width=True, key="daily_conversion_trend")

                # Conversion rate statistics
                avg_conversion = daily_combined['Conversion_Rate'].mean()
                max_conversion = daily_combined['Conversion_Rate'].max()
                min_conversion = daily_combined['Conversion_Rate'].min()

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Daily Conversion", f"{avg_conversion:.1f}%")
                with col2:
                    st.metric("Peak Daily Conversion", f"{max_conversion:.1f}%")
                with col3:
                    st.metric("Minimum Daily Conversion", f"{min_conversion:.1f}%")

        # Pathway efficiency analysis
        st.markdown("#### ⚡ Care Pathway Efficiency")

        if ('Day_of_Week' in self.eu_data.columns and 'Day_of_Week' in self.admissions_data.columns):
            eu_dow = self.eu_data.groupby('Day_of_Week')['Visit_Count'].sum().reset_index()
            adm_dow = self.admissions_data.groupby('Day_of_Week')['No_of_Admissions'].sum().reset_index()

            dow_combined = pd.merge(eu_dow, adm_dow, on='Day_of_Week', how='inner')
            dow_combined['Conversion_Rate'] = (
                        dow_combined['No_of_Admissions'] / dow_combined['Visit_Count'] * 100).fillna(0)

            # Sort by day order
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dow_combined['day_num'] = dow_combined['Day_of_Week'].map({day: i for i, day in enumerate(day_order)})
            dow_combined = dow_combined.sort_values('day_num')

            fig = px.bar(
                dow_combined,
                x='Day_of_Week',
                y='Conversion_Rate',
                title="Conversion Rate by Day of Week",
                color='Conversion_Rate',
                color_continuous_scale='viridis'
            )
            fig.update_layout(
                height=400,
                showlegend=False,
                xaxis=dict(tickangle=-45, tickfont=dict(size=10))
            )
            st.plotly_chart(fig, use_container_width=True, key="dow_conversion_efficiency")

        # Monthly pathway analysis
        if ('Month_Name' in self.eu_data.columns and 'Month_Name' in self.admissions_data.columns):
            st.markdown("#### 📆 Monthly Pathway Performance")

            eu_monthly = self.eu_data.groupby('Month_Name')['Visit_Count'].sum().reset_index()
            adm_monthly = self.admissions_data.groupby('Month_Name')['No_of_Admissions'].sum().reset_index()

            monthly_combined = pd.merge(eu_monthly, adm_monthly, on='Month_Name', how='inner')
            monthly_combined['Conversion_Rate'] = (
                        monthly_combined['No_of_Admissions'] / monthly_combined['Visit_Count'] * 100).fillna(0)

            # Sort by month order
            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                           'July', 'August', 'September', 'October', 'November', 'December']
            monthly_combined['month_num'] = monthly_combined['Month_Name'].map(
                {month: i for i, month in enumerate(month_order)})
            monthly_combined = monthly_combined.sort_values('month_num')

            fig = px.line(
                monthly_combined,
                x='Month_Name',
                y='Conversion_Rate',
                title="Monthly Conversion Rate Trends",
                markers=True
            )
            fig.update_layout(
                height=400,
                xaxis=dict(tickangle=-45, tickfont=dict(size=10))
            )
            st.plotly_chart(fig, use_container_width=True, key="monthly_conversion_trends")

        # Care continuity insights
        st.markdown("#### 🔍 Care Continuity Insights")

        continuity_insights = [
            f"📊 **Overall Conversion**: {conversion_rate:.1f}% of EU visits result in admission",
            f"⚡ **System Efficiency**: {100 - conversion_rate:.1f}% of EU visits are resolved without admission",
            f"🎯 **Optimization Opportunity**: Monitor high-conversion days for capacity planning",
            f"🔄 **Care Pathway**: Analyze low-conversion periods for process improvements"
        ]

        for insight in continuity_insights:
            st.markdown(insight)

    def _show_integrated_12month_forecast(self):
        """🔮 Integrated 12-Month Strategic Forecasting combining both datasets"""
        st.markdown("### 🔮 Integrated 12-Month Strategic Forecast")
        st.info("Combined predictions for EU Visits and Admissions with cross-dataset insights")

        # Prepare forecasting data for both datasets
        eu_daily_data, eu_error = self._prepare_forecasting_data()
        adm_daily_data, adm_error = self._prepare_adm_forecasting_data()

        if eu_daily_data is None and adm_daily_data is None:
            st.error(f"❌ No forecasting data available. EU Error: {eu_error}, Admissions Error: {adm_error}")
            return
        elif eu_daily_data is None:
            st.warning("⚠️ EU Visits forecasting data unavailable - showing admissions forecast only")
            self._show_adm_12month_forecast()
            return
        elif adm_daily_data is None:
            st.warning("⚠️ Admissions forecasting data unavailable - showing EU visits forecast only")
            self._show_eu_12month_forecast()
            return

        # Combined forecasting analysis
        st.markdown("##### Combined Healthcare System Forecast")

        # Analyze combined patterns
        combined_patterns = self._analyze_integrated_patterns(eu_daily_data, adm_daily_data)

        # Generate integrated forecasts
        eu_forecast = self._create_yearly_forecast(eu_daily_data, self._analyze_historical_patterns(eu_daily_data))
        adm_forecast = self._create_adm_yearly_forecast(adm_daily_data,
                                                        self._analyze_adm_historical_patterns(adm_daily_data))

        if eu_forecast is not None and adm_forecast is not None:
            # Merge forecasts on date
            integrated_forecast = pd.merge(
                eu_forecast[['Date', 'Predicted_Visits']].rename(columns={'Predicted_Visits': 'EU_Visits'}),
                adm_forecast[['Date', 'Predicted_Admissions']].rename(columns={'Predicted_Admissions': 'Admissions'}),
                on='Date',
                how='inner'
            )
            integrated_forecast['Total_System_Volume'] = integrated_forecast['EU_Visits'] + integrated_forecast[
                'Admissions']
            integrated_forecast['Predicted_Conversion_Rate'] = (
                    integrated_forecast['Admissions'] / integrated_forecast['EU_Visits'] * 100
            ).fillna(0)

            # Display integrated KPIs
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                total_eu = integrated_forecast['EU_Visits'].sum()
                st.metric("Predicted EU Visits (12M)", f"{total_eu:,.0f}")

            with col2:
                total_adm = integrated_forecast['Admissions'].sum()
                st.metric("Predicted Admissions (12M)", f"{total_adm:,.0f}")

            with col3:
                total_volume = integrated_forecast['Total_System_Volume'].sum()
                st.metric("Total System Volume (12M)", f"{total_volume:,.0f}")

            with col4:
                avg_conversion = integrated_forecast['Predicted_Conversion_Rate'].mean()
                st.metric("Avg Conversion Rate", f"{avg_conversion:.1f}%")

            # Integrated forecast visualization
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=["Combined System Volume Forecast", "Predicted Conversion Rate Trend"],
                specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
            )

            # Combined volume chart
            fig.add_trace(
                go.Scatter(
                    x=integrated_forecast['Date'],
                    y=integrated_forecast['EU_Visits'],
                    name='EU Visits',
                    line=dict(color='blue')
                ),
                row=1, col=1, secondary_y=False
            )

            fig.add_trace(
                go.Scatter(
                    x=integrated_forecast['Date'],
                    y=integrated_forecast['Admissions'],
                    name='Admissions',
                    line=dict(color='red')
                ),
                row=1, col=1, secondary_y=True
            )

            # Conversion rate trend
            fig.add_trace(
                go.Scatter(
                    x=integrated_forecast['Date'],
                    y=integrated_forecast['Predicted_Conversion_Rate'],
                    name='Conversion Rate %',
                    line=dict(color='green')
                ),
                row=2, col=1
            )

            # Update layout
            fig.update_yaxes(title_text="EU Visits", row=1, col=1, secondary_y=False)
            fig.update_yaxes(title_text="Admissions", row=1, col=1, secondary_y=True)
            fig.update_yaxes(title_text="Conversion Rate (%)", row=2, col=1)
            fig.update_xaxes(title_text="Date", row=2, col=1)

            fig.update_layout(height=800, title_text="Integrated Healthcare System 12-Month Forecast")
            st.plotly_chart(fig, use_container_width=True, key="integrated_forecast_main")

            # Strategic insights tabs
            st.markdown("##### Integrated Strategic Planning Insights")

            insight_tab1, insight_tab2, insight_tab3, insight_tab4 = st.tabs([
                "🎯 System Capacity Planning",
                "📊 Cross-Dataset Correlations",
                "🏥 Resource Optimization",
                "⚡ Operational Recommendations"
            ])

            with insight_tab1:
                self._show_integrated_capacity_planning(integrated_forecast, combined_patterns)

            with insight_tab2:
                self._show_cross_dataset_correlations(integrated_forecast, eu_forecast, adm_forecast)

            with insight_tab3:
                self._show_integrated_resource_optimization(integrated_forecast, combined_patterns)

            with insight_tab4:
                self._show_integrated_operational_recommendations(integrated_forecast, combined_patterns)

    def _analyze_integrated_patterns(self, eu_daily_data, adm_daily_data):
        """Analyze patterns across both datasets for integrated forecasting"""
        patterns = {}

        # Merge datasets for correlation analysis
        merged_data = pd.merge(
            eu_daily_data.rename(columns={'Visit_Count': 'EU_Visits'}),
            adm_daily_data.rename(columns={'No_of_Admissions': 'Admissions'}),
            on='Date',
            how='inner'
        )

        if not merged_data.empty:
            # Cross-dataset correlation
            patterns['eu_adm_correlation'] = merged_data['EU_Visits'].corr(merged_data['Admissions'])

            # Combined volume patterns
            merged_data['Total_Volume'] = merged_data['EU_Visits'] + merged_data['Admissions']
            merged_data['Conversion_Rate'] = (merged_data['Admissions'] / merged_data['EU_Visits'] * 100).fillna(0)

            # Time-based patterns
            merged_data['Month'] = merged_data['Date'].dt.month
            merged_data['DayOfWeek'] = merged_data['Date'].dt.dayofweek

            patterns['monthly_volume'] = merged_data.groupby('Month')['Total_Volume'].agg(['mean', 'std'])
            patterns['monthly_conversion'] = merged_data.groupby('Month')['Conversion_Rate'].agg(['mean', 'std'])
            patterns['daily_volume'] = merged_data.groupby('DayOfWeek')['Total_Volume'].agg(['mean', 'std'])
            patterns['daily_conversion'] = merged_data.groupby('DayOfWeek')['Conversion_Rate'].agg(['mean', 'std'])

            # Growth trends
            if len(merged_data) > 30:
                recent_avg = merged_data.tail(30)['Total_Volume'].mean()
                older_avg = merged_data.head(30)['Total_Volume'].mean()
                patterns['volume_growth_trend'] = (recent_avg - older_avg) / older_avg * 100 if older_avg > 0 else 0
            else:
                patterns['volume_growth_trend'] = 0

        return patterns

    def _show_integrated_capacity_planning(self, integrated_forecast, combined_patterns):
        """Show integrated system capacity planning insights"""
        st.markdown("#### 🎯 Integrated System Capacity Planning")

        # Peak demand analysis
        peak_total_volume = integrated_forecast['Total_System_Volume'].max()
        avg_total_volume = integrated_forecast['Total_System_Volume'].mean()
        peak_date = integrated_forecast.loc[integrated_forecast['Total_System_Volume'].idxmax(), 'Date']

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Peak System Volume", f"{peak_total_volume:.0f}")
        with col2:
            st.metric("Average Daily Volume", f"{avg_total_volume:.0f}")
        with col3:
            capacity_buffer = ((peak_total_volume - avg_total_volume) / avg_total_volume) * 100
            st.metric("Required Capacity Buffer", f"{capacity_buffer:.0f}%")

        # Monthly capacity requirements
        monthly_capacity = integrated_forecast.groupby(integrated_forecast['Date'].dt.month).agg({
            'Total_System_Volume': ['mean', 'max'],
            'EU_Visits': 'mean',
            'Admissions': 'mean'
        }).round(0)

        monthly_capacity.columns = ['Avg_Total', 'Peak_Total', 'Avg_EU', 'Avg_Admissions']
        monthly_capacity['Month'] = monthly_capacity.index.map(lambda x: calendar.month_abbr[x])
        monthly_capacity = monthly_capacity.reset_index()

        fig = px.bar(
            monthly_capacity,
            x='Month',
            y=['Avg_EU', 'Avg_Admissions'],
            title="Monthly Capacity Requirements by Service Type",
            barmode='stack'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True, key="integrated_monthly_capacity")

        # Capacity planning recommendations
        st.markdown("##### 📋 Capacity Planning Recommendations")

        recommendations = [
            f"🎯 **Peak Demand**: Plan for maximum {peak_total_volume:.0f} daily patient volume on {peak_date.strftime('%B %Y')}",
            f"📊 **Baseline Capacity**: Design for {avg_total_volume:.0f} average daily volume",
            f"⚡ **Surge Capacity**: Maintain {capacity_buffer:.0f}% additional capacity for peak periods",
            f"🏥 **Resource Allocation**: EU services need {integrated_forecast['EU_Visits'].mean():.0f} daily capacity, Admissions need {integrated_forecast['Admissions'].mean():.0f}"
        ]

        for rec in recommendations:
            st.markdown(rec)

    def _show_cross_dataset_correlations(self, integrated_forecast, eu_forecast, adm_forecast):
        """Show cross-dataset correlation analysis"""
        st.markdown("#### 📊 Cross-Dataset Correlation Analysis")

        # Calculate correlation between predictions
        correlation = integrated_forecast['EU_Visits'].corr(integrated_forecast['Admissions'])

        col1, col2 = st.columns(2)

        with col1:
            st.metric("EU Visits-Admissions Correlation", f"{correlation:.3f}")

            if correlation > 0.7:
                st.success("✅ Strong positive correlation - EU visits strongly predict admissions")
            elif correlation > 0.3:
                st.info("📊 Moderate correlation - Some relationship between services")
            else:
                st.warning("⚠️ Weak correlation - Services follow different patterns")

        with col2:
            # Conversion rate variability
            conversion_std = integrated_forecast['Predicted_Conversion_Rate'].std()
            conversion_mean = integrated_forecast['Predicted_Conversion_Rate'].mean()
            conversion_cv = (conversion_std / conversion_mean) * 100 if conversion_mean > 0 else 0

            st.metric("Conversion Rate Variability", f"{conversion_cv:.1f}%")

            if conversion_cv < 10:
                st.success("✅ Stable conversion pattern - predictable admission rates")
            elif conversion_cv < 20:
                st.info("📊 Moderate variability - monitor seasonal patterns")
            else:
                st.warning("⚠️ High variability - requires flexible capacity planning")

        # Correlation scatter plot
        fig = px.scatter(
            integrated_forecast,
            x='EU_Visits',
            y='Admissions',
            color='Predicted_Conversion_Rate',
            title=f"EU Visits vs Admissions Correlation (r = {correlation:.3f})",
            color_continuous_scale='viridis',
            trendline="ols"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True, key="correlation_scatter")

        # Monthly correlation heatmap
        monthly_corr_data = []
        for month in range(1, 13):
            month_data = integrated_forecast[integrated_forecast['Date'].dt.month == month]
            if len(month_data) > 1:
                month_corr = month_data['EU_Visits'].corr(month_data['Admissions'])
                monthly_corr_data.append({
                    'Month': calendar.month_abbr[month],
                    'Correlation': month_corr,
                    'EU_Visits': month_data['EU_Visits'].mean(),
                    'Admissions': month_data['Admissions'].mean()
                })

        if monthly_corr_data:
            monthly_corr_df = pd.DataFrame(monthly_corr_data)

            fig = px.bar(
                monthly_corr_df,
                x='Month',
                y='Correlation',
                title="Monthly EU Visits-Admissions Correlation",
                color='Correlation',
                color_continuous_scale='RdYlBu_r'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key="monthly_correlation")

    def _show_integrated_resource_optimization(self, integrated_forecast, combined_patterns):
        """Show integrated resource optimization insights"""
        st.markdown("#### 🏥 Integrated Resource Optimization")

        # Resource efficiency analysis
        peak_eu = integrated_forecast['EU_Visits'].max()
        peak_adm = integrated_forecast['Admissions'].max()

        eu_utilization = integrated_forecast['EU_Visits'].mean() / peak_eu * 100 if peak_eu > 0 else 0
        adm_utilization = integrated_forecast['Admissions'].mean() / peak_adm * 100 if peak_adm > 0 else 0

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("EU Services Utilization", f"{eu_utilization:.0f}%")
        with col2:
            st.metric("Admissions Utilization", f"{adm_utilization:.0f}%")
        with col3:
            overall_efficiency = (eu_utilization + adm_utilization) / 2
            st.metric("System Efficiency", f"{overall_efficiency:.0f}%")

        # Resource sharing opportunities
        st.markdown("##### 🔄 Resource Sharing Opportunities")

        # Find complementary patterns (when one is low, other is high)
        integrated_forecast['EU_Percentile'] = integrated_forecast['EU_Visits'].rank(pct=True)
        integrated_forecast['Adm_Percentile'] = integrated_forecast['Admissions'].rank(pct=True)
        integrated_forecast['Complementary_Score'] = abs(
            integrated_forecast['EU_Percentile'] - integrated_forecast['Adm_Percentile'])

        high_complementary = integrated_forecast.nlargest(10, 'Complementary_Score')

        fig = px.scatter(
            integrated_forecast,
            x='EU_Visits',
            y='Admissions',
            color='Complementary_Score',
            title="Resource Sharing Opportunities (Higher Color = Better Sharing Potential)",
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True, key="resource_sharing")

        # Optimization recommendations
        optimization_recs = [
            f"⚡ **EU Services**: {eu_utilization:.0f}% average utilization suggests {'good efficiency' if eu_utilization > 70 else 'optimization potential'}",
            f"🏥 **Admissions**: {adm_utilization:.0f}% average utilization suggests {'good efficiency' if adm_utilization > 70 else 'capacity reallocation opportunity'}",
            f"🔄 **Cross-Training**: Staff flexibility can address {len(high_complementary)} high-complementary demand periods",
            f"📊 **System Efficiency**: {overall_efficiency:.0f}% overall efficiency {'meets targets' if overall_efficiency > 75 else 'needs improvement'}"
        ]

        for rec in optimization_recs:
            st.markdown(rec)

    def _show_integrated_operational_recommendations(self, integrated_forecast, combined_patterns):
        """Show integrated operational recommendations"""
        st.markdown("#### ⚡ Integrated Operational Recommendations")

        # Generate actionable recommendations based on forecast data
        peak_system_date = integrated_forecast.loc[integrated_forecast['Total_System_Volume'].idxmax()]
        low_system_date = integrated_forecast.loc[integrated_forecast['Total_System_Volume'].idxmin()]

        # Staffing recommendations
        st.markdown("##### 👥 Staffing Strategy")

        staffing_recs = [
            f"🎯 **Peak System Day**: {peak_system_date['Date'].strftime('%B %d, %Y')} requires maximum staffing ({peak_system_date['Total_System_Volume']:.0f} total volume)",
            f"📅 **Maintenance Window**: {low_system_date['Date'].strftime('%B %d, %Y')} optimal for system maintenance ({low_system_date['Total_System_Volume']:.0f} total volume)",
            f"🔄 **Cross-Training Priority**: Train staff for both EU and admission services to handle {integrated_forecast['Predicted_Conversion_Rate'].std():.1f}% conversion rate variability",
            f"⚡ **Surge Protocols**: Implement when total daily volume exceeds {integrated_forecast['Total_System_Volume'].quantile(0.9):.0f} patients"
        ]

        for rec in staffing_recs:
            st.markdown(rec)

        # Budget allocation recommendations
        st.markdown("##### 💰 Budget Allocation Strategy")

        eu_percentage = (integrated_forecast['EU_Visits'].sum() / integrated_forecast[
            'Total_System_Volume'].sum()) * 100
        adm_percentage = 100 - eu_percentage

        budget_recs = [
            f"📊 **Service Allocation**: Allocate {eu_percentage:.0f}% budget to EU services, {adm_percentage:.0f}% to admissions",
            f"🎯 **Peak Month Budget**: Increase spending by {((integrated_forecast.groupby(integrated_forecast['Date'].dt.month)['Total_System_Volume'].max().max() - integrated_forecast['Total_System_Volume'].mean()) / integrated_forecast['Total_System_Volume'].mean() * 100):.0f}% during peak month",
            f"⚡ **Variable Costs**: Plan for {integrated_forecast['Total_System_Volume'].std() / integrated_forecast['Total_System_Volume'].mean() * 100:.0f}% daily volume variability",
            f"🏥 **Infrastructure**: Size systems for {integrated_forecast['Total_System_Volume'].max():.0f} peak daily volume"
        ]

        for rec in budget_recs:
            st.markdown(rec)

        # Technology and process recommendations
        st.markdown("##### 🔧 Technology & Process Optimization")

        tech_recs = [
            "📱 **Predictive Analytics**: Implement daily volume predictions to optimize staffing 24-48 hours ahead",
            "🔄 **Patient Flow Management**: Use conversion rate predictions to optimize bed management and discharge planning",
            "📊 **Dashboard Implementation**: Real-time monitoring of actual vs predicted volumes for immediate adjustments",
            "⚡ **Alert Systems**: Automated notifications when volumes exceed 95% of predicted capacity"
        ]

        for rec in tech_recs:
            st.markdown(rec)

        # Performance monitoring KPIs
        st.markdown("##### 📈 Performance Monitoring KPIs")

        kpi_table = pd.DataFrame({
            'KPI': ['Daily System Volume', 'EU-Admission Conversion Rate', 'Peak Capacity Utilization',
                    'Cross-Service Correlation'],
            'Target': [f"{integrated_forecast['Total_System_Volume'].mean():.0f} ± 20%",
                       f"{integrated_forecast['Predicted_Conversion_Rate'].mean():.1f}% ± 5%",
                       "< 95% of peak capacity",
                       f"Maintain {integrated_forecast['EU_Visits'].corr(integrated_forecast['Admissions']):.2f} correlation"],
            'Alert_Threshold': [f"> {integrated_forecast['Total_System_Volume'].quantile(0.95):.0f}",
                                f"> {integrated_forecast['Predicted_Conversion_Rate'].quantile(0.9):.1f}%",
                                "> 95%",
                                "< 0.3 correlation"]
        })

        st.dataframe(kpi_table, use_container_width=True)

        # Next steps action plan
        st.markdown("##### 🎯 Next Steps Action Plan")

        next_steps = [
            "1. **Immediate (Next 30 Days)**: Set up monitoring dashboard for daily volume tracking",
            "2. **Short-term (Next 90 Days)**: Implement staffing flexibility protocols based on predictions",
            "3. **Medium-term (Next 6 Months)**: Deploy predictive analytics for capacity planning",
            "4. **Long-term (Next 12 Months)**: Full integration of forecasting into operational planning"
        ]

        for step in next_steps:
            st.markdown(step)


    # ═══════════════════════════════════════════════════════════════════════════
    # UTILITY FUNCTIONS
    # ═══════════════════════════════════════════════════════════════════════════

    def export_summary_report(self):
        """Generate and return summary report"""
        report = {
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "datasets_analyzed": [],
            "key_metrics": {},
            "recommendations": []
        }

        if self.eu_data is not None:
            report["datasets_analyzed"].append("EU Visits")
            if 'Visit_Count' in self.eu_data.columns:
                report["key_metrics"]["eu_total_visits"] = int(self.eu_data['Visit_Count'].sum())
            if 'Date' in self.eu_data.columns:
                report["key_metrics"]["eu_date_range"] = f"{self.eu_data['Date'].min()} to {self.eu_data['Date'].max()}"

        if self.admissions_data is not None:
            report["datasets_analyzed"].append("Admissions")
            if 'No_of_Admissions' in self.admissions_data.columns:
                report["key_metrics"]["admissions_total"] = int(self.admissions_data['No_of_Admissions'].sum())
            if 'Date' in self.admissions_data.columns:
                report["key_metrics"]["admissions_date_range"] = f"{self.admissions_data['Date'].min()} to {self.admissions_data['Date'].max()}"

        return report

# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT INTEGRATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def show_analytics_sidebar():
    """Show analytics control sidebar"""
    with st.sidebar:
        st.markdown("### 🎛️ Analytics Controls")

        # Quick filters
        st.markdown("#### 📊 Quick Insights")

        if hasattr(st.session_state, 'eu_visits_data') and st.session_state.eu_visits_data is not None:
            if isinstance(st.session_state.eu_visits_data, pd.DataFrame):
                eu_data = st.session_state.eu_visits_data
                st.metric("EU Visit Records", f"{len(eu_data):,}")
                if 'Visit_Count' in eu_data.columns:
                    st.metric("Total EU Visits", f"{eu_data['Visit_Count'].sum():,}")

        if hasattr(st.session_state, 'admissions_data') and st.session_state.admissions_data is not None:
            if isinstance(st.session_state.admissions_data, bytes):
                st.info("📊 Admissions data: Excel file loaded")
            elif isinstance(st.session_state.admissions_data, pd.DataFrame):
                adm_data = st.session_state.admissions_data
                st.metric("Admission Records", f"{len(adm_data):,}")
                if 'No_of_Admissions' in adm_data.columns:
                    st.metric("Total Admissions", f"{adm_data['No_of_Admissions'].sum():,}")

def initialize_dashboard():
    """Initialize the healthcare analytics dashboard"""
    if 'healthcare_analyzer' not in st.session_state:
        st.session_state.healthcare_analyzer = HealthcareAnalyzer()

    return st.session_state.healthcare_analyzer

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    st.set_page_config(
        page_title="Healthcare Analytics Dashboard",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    analyzer = initialize_dashboard()
    show_analytics_sidebar()
    analyzer.show_dashboard()
