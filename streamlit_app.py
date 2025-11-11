# streamlit_app.py - Healthcare Data Processor - IMPROVED VERSION
# Key improvements: Better error handling, performance optimization, UI enhancements

import streamlit as st

# Configure page FIRST - before any other Streamlit commands
st.set_page_config(
    page_title="Reduced Demand of Emergency Unit",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Core imports
import tempfile
import os
import shutil
import time
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import traceback
import io
import base64
import threading
import queue

# Enhanced imports for visualization support
try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio

    PLOTLY_AVAILABLE = True
    pio.templates.default = "plotly_white"
except ImportError:
    PLOTLY_AVAILABLE = False


def safe_date_parsing(data, date_column='Date_String', possible_formats=None):
    """Safely parse dates with multiple format attempts"""
    if possible_formats is None:
        possible_formats = [
            '%d/%m/%Y', '%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y', '%Y/%m/%d',
            '%d/%m/%y', '%y-%m-%d', '%m/%d/%y', '%d-%m-%y', '%y/%m/%d'
        ]

    if date_column not in data.columns:
        return None, f"Column '{date_column}' not found"

    parsed_dates = None
    successful_format = None

    for date_format in possible_formats:
        try:
            parsed_dates = pd.to_datetime(data[date_column], format=date_format, errors='coerce')
            valid_count = parsed_dates.notna().sum()
            total_count = len(data)

            # If more than 80% of dates parsed successfully, use this format
            if valid_count > 0 and (valid_count / total_count) > 0.8:
                successful_format = date_format
                break
        except Exception:
            continue

    # Fallback to pandas automatic parsing
    if parsed_dates is None or parsed_dates.notna().sum() == 0:
        try:
            parsed_dates = pd.to_datetime(data[date_column], errors='coerce')
            successful_format = "automatic"
        except Exception as e:
            return None, f"Could not parse dates: {str(e)}"

    return parsed_dates, successful_format


def memory_efficient_csv_download(dataframe, chunk_size=10000):
    """Create memory-efficient CSV download for large DataFrames"""
    if len(dataframe) <= chunk_size:
        # Small DataFrame, process normally
        return dataframe.to_csv(index=False)
    else:
        # Large DataFrame, use chunked processing
        output = io.StringIO()

        # Write header
        dataframe.head(0).to_csv(output, index=False)

        # Write data in chunks
        for start_idx in range(0, len(dataframe), chunk_size):
            end_idx = min(start_idx + chunk_size, len(dataframe))
            chunk = dataframe.iloc[start_idx:end_idx]
            chunk.to_csv(output, header=False, index=False, mode='a')

        csv_data = output.getvalue()
        output.close()
        return csv_data


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SESSION STATE MANAGEMENT - IMPROVED WITH BETTER PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════════

class SessionStateManager:
    """Enhanced session state management with better error handling"""

    @staticmethod
    def initialize_session_state():
        """Initialize session state with validation"""
        defaults = {
            'processing_complete': False,
            'eu_visits_data': None,
            'eu_visits_filename': None,
            'admissions_data': None,
            'admissions_filename': None,
            'processing_config': {},
            'processing_timestamp': None,
            'uploaded_files_info': [],
            'show_visualizations': False,
            'viz_loading': False,
            'viz_data_processed': False,
            'viz_module_available': None,
            'first_visit': True,
            'analytics_promoted': False,
            'processing_start_time': None,
            'processing_duration': None,
            'error_messages': [],
            'last_activity': None,
            'session_id': None
        }

        # Initialize or validate existing state
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

        # Generate session ID if missing
        if st.session_state.session_id is None:
            st.session_state.session_id = f"session_{int(time.time())}"

        # Update activity timestamp
        st.session_state.last_activity = datetime.now()

    @staticmethod
    def save_processing_results(eu_data=None, eu_filename=None, admissions_data=None,
                                admissions_filename=None, config=None, files_info=None):
        """Save processing results with validation"""
        try:
            st.session_state.processing_complete = True
            st.session_state.processing_timestamp = datetime.now()
            st.session_state.last_activity = datetime.now()

            if eu_data is not None:
                # Validate EU data
                if isinstance(eu_data, pd.DataFrame) and not eu_data.empty:
                    st.session_state.eu_visits_data = eu_data
                    st.session_state.eu_visits_filename = eu_filename
                else:
                    st.session_state.error_messages.append("EU visits data is empty or invalid")

            if admissions_data is not None:
                # Validate admissions data
                if isinstance(admissions_data, bytes) and len(admissions_data) > 0:
                    st.session_state.admissions_data = admissions_data
                    st.session_state.admissions_filename = admissions_filename
                else:
                    st.session_state.error_messages.append("Admissions data is empty or invalid")

            if config is not None:
                st.session_state.processing_config = config

            if files_info is not None:
                st.session_state.uploaded_files_info = files_info

            # Calculate processing duration
            if st.session_state.processing_start_time is not None:
                duration_seconds = (datetime.now() - st.session_state.processing_start_time).total_seconds()
                st.session_state.processing_duration = duration_seconds / 60

        except Exception as e:
            st.session_state.error_messages.append(f"Error saving results: {str(e)}")

    @staticmethod
    def clear_results():
        """Clear processing results with confirmation"""
        keys_to_clear = [
            'processing_complete', 'eu_visits_data', 'eu_visits_filename',
            'admissions_data', 'admissions_filename', 'processing_config',
            'processing_timestamp', 'uploaded_files_info', 'show_visualizations',
            'viz_loading', 'viz_data_processed', 'processing_start_time',
            'processing_duration', 'error_messages'
        ]

        for key in keys_to_clear:
            if key in st.session_state:
                if key in ['processing_complete', 'show_visualizations', 'viz_loading', 'viz_data_processed']:
                    st.session_state[key] = False
                elif key in ['processing_config', 'uploaded_files_info', 'error_messages']:
                    st.session_state[key] = {} if key == 'processing_config' else []
                else:
                    st.session_state[key] = None

        st.session_state.last_activity = datetime.now()

    @staticmethod
    def is_session_expired(max_minutes=60):
        """Check if session has expired with better logic"""
        if st.session_state.processing_timestamp is None:
            return True

        elapsed = datetime.now() - st.session_state.processing_timestamp
        return elapsed.total_seconds() > (max_minutes * 60)

    @staticmethod
    def get_time_remaining(max_minutes=60):
        """Get time remaining with validation"""
        if st.session_state.processing_timestamp is None:
            return 0

        elapsed = datetime.now() - st.session_state.processing_timestamp
        remaining_seconds = (max_minutes * 60) - elapsed.total_seconds()
        return max(0, remaining_seconds)

    @staticmethod
    def has_valid_results():
        """Check if we have valid results to display"""
        return (st.session_state.processing_complete and
                not SessionStateManager.is_session_expired() and
                (st.session_state.eu_visits_data is not None or
                 st.session_state.admissions_data is not None))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. IMPROVED MODULE CAPABILITIES DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

class ModuleCapabilities:
    """Enhanced module detection with caching"""

    def __init__(self):
        # Initialize all capabilities
        self.eu_visits_available = False
        self.eu_month_filtering_available = False
        self.eu_day_filtering_available = False
        self.admissions_available = False
        self.month_filtering_available = False
        self.day_filtering_available = False
        self.visualizations_available = False
        self.detection_errors = []

        self._detect_capabilities()

    def _detect_capabilities(self):
        """Enhanced capability detection with better error handling"""
        # EU_Visits module detection
        try:
            from EU_Visits import (
                process_multiple_files_with_dynamic_detection,
                process_multiple_files_with_month_filter,
                process_multiple_files_with_day_filter,
            )
            self.eu_visits_available = True
            self.eu_month_filtering_available = True
            self.eu_day_filtering_available = True
        except ImportError as e:
            try:
                from EU_Visits import (
                    process_multiple_files_with_dynamic_detection,
                    process_multiple_files_with_month_filter,
                )
                self.eu_visits_available = True
                self.eu_month_filtering_available = True
                self.eu_day_filtering_available = False
                self.detection_errors.append(f"EU day filtering not available: {str(e)}")
            except ImportError as e:
                try:
                    from EU_Visits import process_multiple_files_with_dynamic_detection
                    self.eu_visits_available = True
                    self.eu_month_filtering_available = False
                    self.eu_day_filtering_available = False
                    self.detection_errors.append(f"EU advanced filtering not available: {str(e)}")
                except ImportError as e:
                    self.eu_visits_available = False
                    self.detection_errors.append(f"EU_Visits module not available: {str(e)}")

        # Core module detection
        try:
            from core import (
                main as admissions_main,
                main_with_month_filter,
                main_with_day_filter,
                create_month_filter,
                create_day_filter,
                MONTH_NUMBER_TO_ABBR,
            )
            self.admissions_available = True
            self.month_filtering_available = True
            self.day_filtering_available = True
        except ImportError as e:
            try:
                from core import (
                    main as admissions_main,
                    main_with_month_filter,
                    create_month_filter,
                    MONTH_NUMBER_TO_ABBR,
                )
                self.admissions_available = True
                self.month_filtering_available = True
                self.day_filtering_available = False
                self.detection_errors.append(f"Day filtering not available: {str(e)}")
            except ImportError as e:
                try:
                    from core import main as admissions_main
                    self.admissions_available = True
                    self.month_filtering_available = False
                    self.day_filtering_available = False
                    self.detection_errors.append(f"Advanced filtering not available: {str(e)}")
                except ImportError as e:
                    self.admissions_available = False
                    self.detection_errors.append(f"Core admissions module not available: {str(e)}")

        # Visualizations module detection
        try:
            import visualizations
            self.visualizations_available = True
            st.session_state.viz_module_available = True
        except ImportError as e:
            self.visualizations_available = False
            st.session_state.viz_module_available = False
            self.detection_errors.append(f"Visualizations module not available: {str(e)}")

    

    def get_available_processing_options(self):
        """Get available processing options based on capabilities"""
        options = []
        if self.eu_visits_available:
            options.append("Only EU Visits")
        if self.admissions_available:
            options.append("Only Admissions")
        if self.eu_visits_available and self.admissions_available:
            options.append("Both")
        return options if options else ["No modules available"]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ENHANCED UI HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

class UIHelpers:
    """Enhanced UI helper functions"""

    @staticmethod
    def show_error_message(title: str, message: str, expandable_details: str = None):
        """Standardized error message display"""
        st.error(f"❌ **{title}**")
        st.error(message)

        if expandable_details:
            with st.expander("🔍 Technical Details"):
                st.code(expandable_details)

    @staticmethod
    def show_success_message(title: str, message: str, metrics: Dict[str, str] = None):
        """Standardized success message display"""
        st.success(f"✅ **{title}**")
        st.success(message)

        if metrics:
            cols = st.columns(len(metrics))
            for i, (key, value) in enumerate(metrics.items()):
                with cols[i]:
                    st.metric(key, value)

    @staticmethod
    def show_progress_indicator(current: int, total: int, message: str):
        """Enhanced progress indicator"""
        progress = current / total if total > 0 else 0
        st.progress(progress)
        st.text(f"{message} ({current}/{total})")

    @staticmethod
    def get_weekday_weekend_count(start_date: date, end_date: date) -> Tuple[int, int]:
        """Calculate weekday and weekend count for a date range"""
        current_date = start_date
        weekday_count = 0
        weekend_count = 0

        while current_date <= end_date:
            if current_date.weekday() < 5:  # Monday = 0, Sunday = 6
                weekday_count += 1
            else:
                weekend_count += 1
            current_date += timedelta(days=1)

        return weekday_count, weekend_count

    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size in human readable format"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"

    @staticmethod
    def estimate_processing_time(uploaded_files, enable_weather: bool = False) -> Tuple[int, int]:
        """Enhanced processing time estimation"""
        base_time = len(uploaded_files) * 15  # 15 seconds per file
        total_size = sum(file.size for file in uploaded_files)

        # Adjust for file size
        if total_size > 10 * 1024 * 1024:  # If total > 10MB
            base_time += 30
        if total_size > 50 * 1024 * 1024:  # If total > 50MB
            base_time += 60

        # Adjust for features
        if enable_weather:
            base_time += 60  # Add 60 seconds for weather data

        min_time = base_time
        max_time = base_time + 90  # Add buffer for network delays

        return min_time, max_time

    @staticmethod
    def validate_uploaded_files(uploaded_files) -> Tuple[bool, List[str]]:
        """Validate uploaded files"""
        if not uploaded_files:
            return False, ["No files uploaded"]

        errors = []
        valid_extensions = ['.xlsx', '.xls']
        max_file_size = 100 * 1024 * 1024  # 100MB

        for file in uploaded_files:
            # Check extension
            file_ext = os.path.splitext(file.name)[1].lower()
            if file_ext not in valid_extensions:
                errors.append(f"Invalid file type: {file.name} (must be .xlsx or .xls)")

            # Check file size
            if file.size > max_file_size:
                errors.append(f"File too large: {file.name} (max 100MB)")

            # Check if file is empty
            if file.size == 0:
                errors.append(f"Empty file: {file.name}")

        return len(errors) == 0, errors


# ═══════════════════════════════════════════════════════════════════════════════
# 4. IMPROVED FILTER COMPONENTS WITH VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

class FilterComponents:
    """Enhanced filter components with validation"""

    def __init__(self, capabilities: ModuleCapabilities):
        self.capabilities = capabilities

    def get_filter_options(self) -> List[str]:
        """Get available filter options based on capabilities"""
        options = ["No Filter (All Data)", "Year Range"]
        if self.capabilities.month_filtering_available:
            options.append("Month Range")
        if self.capabilities.day_filtering_available:
            options.append("Day Range")
        return options

    def show_year_range_filter(self) -> Optional[Tuple[int, int]]:
        """Enhanced year range filter with validation"""
        col_year1, col_year2 = st.columns(2)

        with col_year1:
            start_year = st.selectbox(
                "Start Year:",
                options=list(range(2015, 2030)),
                index=6,
                help="Select the starting year for data filtering"
            )

        with col_year2:
            end_year = st.selectbox(
                "End Year:",
                options=list(range(2015, 2030)),
                index=7,
                help="Select the ending year for data filtering"
            )

        # Validation
        if start_year > end_year:
            st.error("❌ Start year cannot be greater than end year!")
            return None

        # Show summary
        year_span = end_year - start_year + 1
        st.success(f"📊 Filter: **{start_year} - {end_year}** ({year_span} years)")

        return (start_year, end_year)

    def show_month_range_filter(self) -> Optional[Dict[str, Any]]:
        """Enhanced month range filter with validation"""
        try:
            from core import create_month_filter, MONTH_NUMBER_TO_ABBR

            st.write("**Select Month Range:**")

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.write("**Start Date:**")
                start_month = st.selectbox(
                    "Month:",
                    options=list(range(1, 13)),
                    format_func=lambda x: f"{MONTH_NUMBER_TO_ABBR[x]} ({x:02d})",
                    index=2,
                    key="start_month"
                )
                start_year = st.selectbox(
                    "Year:",
                    options=list(range(2015, 2030)),
                    index=6,
                    key="m_start_year"
                )

            with col_m2:
                st.write("**End Date:**")
                end_month = st.selectbox(
                    "Month:",
                    options=list(range(1, 13)),
                    format_func=lambda x: f"{MONTH_NUMBER_TO_ABBR[x]} ({x:02d})",
                    index=8,
                    key="end_month"
                )
                end_year = st.selectbox(
                    "Year:",
                    options=list(range(2015, 2030)),
                    index=7,
                    key="m_end_year"
                )

            # Create and validate filter
            month_filter = create_month_filter(start_month, start_year, end_month, end_year)

            # Calculate duration
            total_months = (end_year - start_year) * 12 + (end_month - start_month) + 1

            st.success(f"📊 Filter: **{month_filter['description']}** ({total_months} months)")

            return month_filter

        except Exception as e:
            st.error(f"❌ Invalid month range: {str(e)}")
            return None

    def show_day_range_filter(self) -> Optional[Dict[str, Any]]:
        """Enhanced day range filter with validation"""
        try:
            from core import create_day_filter, MONTH_NUMBER_TO_ABBR

            st.write("**Select Date Range:**")

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.write("**Start Date:**")
                start_day = st.selectbox("Day:", options=list(range(1, 32)), index=0, key="d_start_day")
                start_month = st.selectbox(
                    "Month:",
                    options=list(range(1, 13)),
                    format_func=lambda x: f"{MONTH_NUMBER_TO_ABBR[x]} ({x:02d})",
                    index=8,
                    key="d_start_month"
                )
                start_year = st.selectbox("Year:", options=list(range(2015, 2030)), index=6, key="d_start_year")

            with col_d2:
                st.write("**End Date:**")
                end_day = st.selectbox("Day:", options=list(range(1, 32)), index=6, key="d_end_day")
                end_month = st.selectbox(
                    "Month:",
                    options=list(range(1, 13)),
                    format_func=lambda x: f"{MONTH_NUMBER_TO_ABBR[x]} ({x:02d})",
                    index=8,
                    key="d_end_month"
                )
                end_year = st.selectbox("Year:", options=list(range(2015, 2030)), index=6, key="d_end_year")

            # Create and validate filter
            day_filter = create_day_filter(start_day, start_month, start_year, end_day, end_month, end_year)

            st.success(f"📊 Filter: **{day_filter['description']}** ({day_filter['duration_days']} days)")

            return day_filter

        except Exception as e:
            st.error(f"❌ Invalid day range: {str(e)}")
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ENHANCED PROCESSING MANAGER WITH BETTER ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════════════════

class ProcessingManager:
    """Enhanced processing manager with better error handling and progress tracking"""

    def __init__(self, capabilities: ModuleCapabilities):
        self.capabilities = capabilities

    def process_and_save_results(self, temp_file_paths, temp_dir, config, uploaded_files):
        """Enhanced processing with better error handling and progress tracking"""
        # Initialize progress tracking
        progress_container = st.container()
        status_container = st.container()

        # Collect file information for session state
        files_info = [f"{file.name} ({UIHelpers.format_file_size(file.size)})" for file in uploaded_files]

        eu_data = None
        eu_filename = None
        admissions_data = None
        admissions_filename = None
        processing_errors = []

        try:
            # Process EU Visits
            if config['process_option'] in ("Only EU Visits", "Both") and self.capabilities.eu_visits_available:
                with progress_container:
                    st.info(" Processing EU Visits data...")

                try:
                    eu_data = self._process_eu_visits_with_fallback(
                        temp_file_paths, temp_dir, config
                    )

                    if eu_data is not None and not eu_data.empty:
                        eu_filename = self._generate_filename("eu_visits_data", config, "csv")
                        with status_container:
                            UIHelpers.show_success_message(
                                "EU Visits Processing Complete",
                                f"{len(eu_data):,} records processed",
                                {"Records": f"{len(eu_data):,}", "Columns": f"{len(eu_data.columns)}"}
                            )
                    else:
                        processing_errors.append("No EU Visits data found or data is empty")

                except Exception as e:
                    processing_errors.append(f"EU Visits processing failed: {str(e)}")
                    with status_container:
                        UIHelpers.show_error_message(
                            "EU Visits Processing Failed",
                            str(e),
                            traceback.format_exc()
                        )

            # Process Admissions
            if config['process_option'] in ("Only Admissions", "Both") and self.capabilities.admissions_available:
                with progress_container:
                    st.info("Processing Admissions data...")

                try:
                    result_path = self._process_admissions_with_fallback(
                        temp_file_paths, temp_dir, config
                    )

                    if result_path and os.path.exists(result_path):
                        with open(result_path, 'rb') as f:
                            admissions_data = f.read()
                        admissions_filename = self._generate_filename("admissions_data", config, "xlsx")

                        file_size_mb = len(admissions_data) / (1024 * 1024)
                        with status_container:
                            UIHelpers.show_success_message(
                                "Admissions Processing Complete",
                                f"Excel file generated ({file_size_mb:.1f} MB)",
                                {"File Size": f"{file_size_mb:.1f} MB", "Format": "Excel"}
                            )
                    else:
                        processing_errors.append("No Admissions data found or file not generated")

                except Exception as e:
                    processing_errors.append(f"Admissions processing failed: {str(e)}")
                    with status_container:
                        UIHelpers.show_error_message(
                            "Admissions Processing Failed",
                            str(e),
                            traceback.format_exc()
                        )

            # Save results to session state
            SessionStateManager.save_processing_results(
                eu_data=eu_data,
                eu_filename=eu_filename,
                admissions_data=admissions_data,
                admissions_filename=admissions_filename,
                config=config,
                files_info=files_info
            )

            # Add any processing errors to session state
            if processing_errors:
                st.session_state.error_messages.extend(processing_errors)

            return eu_data is not None or admissions_data is not None

        except Exception as e:
            error_msg = f"Critical processing error: {str(e)}"
            processing_errors.append(error_msg)
            st.session_state.error_messages.append(error_msg)

            with status_container:
                UIHelpers.show_error_message(
                    "Processing Failed",
                    error_msg,
                    traceback.format_exc()
                )
            return False

    # Also update the fallback processing methods to reflect that features are always enabled
    def _process_eu_visits_with_fallback(self, temp_file_paths, temp_dir, config):
        """Process EU visits with fallback (features always enabled by default)"""

        # Since features are now mandatory, we start with them enabled
        # but still provide fallback if weather service fails
        fallback_attempts = []

        # Attempt 1: Full features (user expectation since features are mandatory)
        fallback_attempts.append({
            'weather': True,
            'bank_holidays': True,
            'description': 'with all enhanced features'
        })

        # Attempt 2: Without weather only if it fails
        fallback_attempts.append({
            'weather': False,
            'bank_holidays': True,
            'description': 'without weather data (service unavailable)'
        })

        # Attempt 3: Minimal features only if everything fails
        fallback_attempts.append({
            'weather': False,
            'bank_holidays': False,
            'description': 'minimal features only (service fallback)'
        })

        last_error = None
        for attempt in fallback_attempts:
            try:
                if attempt != fallback_attempts[0]:
                    st.warning(f"Retrying EU Visits processing {attempt['description']}")

                return self._process_eu_visits_core(
                    temp_file_paths, temp_dir,
                    config.get('day_filter'),
                    config.get('month_filter'),
                    config.get('year_range'),
                    attempt['bank_holidays'],
                    config.get('uk_region', 'england-and-wales'),
                    attempt['weather']
                )

            except Exception as e:
                last_error = e
                error_msg = str(e).lower()

                # Only continue fallback if it's a network/weather related error
                if any(keyword in error_msg for keyword in ['weather', 'ssl', 'connection', 'timeout', 'network']):
                    continue
                else:
                    raise e

        raise last_error

    def _process_admissions_with_fallback(self, temp_file_paths, temp_dir, config):
        """Process admissions with fallback (weather always enabled by default)"""

        # Since weather is now mandatory, we start with it enabled
        # but still provide fallback if weather service fails
        fallback_attempts = []

        # Attempt 1: With weather (user expectation since feature is mandatory)
        fallback_attempts.append({
            'weather': True,
            'description': 'with weather data'
        })

        # Attempt 2: Without weather only if it fails
        fallback_attempts.append({
            'weather': False,
            'description': 'without weather data (service unavailable)'
        })

        last_error = None
        for attempt in fallback_attempts:
            try:
                if attempt != fallback_attempts[0]:
                    st.warning(f"Retrying Admissions processing {attempt['description']}")

                return self._process_admissions_core(
                    temp_file_paths, temp_dir,
                    config.get('day_filter'),
                    config.get('month_filter'),
                    config.get('year_range'),
                    attempt['weather']
                )

            except Exception as e:
                last_error = e
                error_msg = str(e).lower()

                # Only continue fallback if it's a network/weather related error
                if any(keyword in error_msg for keyword in ['weather', 'ssl', 'connection', 'timeout', 'network']):
                    continue
                else:
                    raise e

        raise last_error

    def _process_eu_visits_core(self, temp_file_paths, temp_dir, day_filter, month_filter, year_range,
                                enable_bank_holidays, uk_region, enable_weather):
        """Core EU Visits processing"""
        if self.capabilities.eu_day_filtering_available and day_filter:
            from EU_Visits import process_multiple_files_with_day_filter
            return process_multiple_files_with_day_filter(
                temp_file_paths, temp_dir,
                enable_bank_holidays=enable_bank_holidays,
                uk_region=uk_region,
                enable_weather=enable_weather,
                day_filter=day_filter
            )
        elif self.capabilities.eu_month_filtering_available and month_filter:
            from EU_Visits import process_multiple_files_with_month_filter
            return process_multiple_files_with_month_filter(
                temp_file_paths, temp_dir,
                enable_bank_holidays=enable_bank_holidays,
                uk_region=uk_region,
                enable_weather=enable_weather,
                month_filter=month_filter
            )
        else:
            from EU_Visits import process_multiple_files_with_dynamic_detection
            return process_multiple_files_with_dynamic_detection(
                temp_file_paths, temp_dir,
                enable_bank_holidays=enable_bank_holidays,
                uk_region=uk_region,
                enable_weather=enable_weather,
                year_filter=year_range
            )

    def _process_admissions_core(self, temp_file_paths, temp_dir, day_filter, month_filter, year_range, enable_weather):
        """Core Admissions processing"""

        def update_progress(step: int, message: str):
            return  # Placeholder for progress updates

        if self.capabilities.day_filtering_available and day_filter:
            from core import main_with_day_filter
            return main_with_day_filter(
                temp_file_paths, temp_dir,
                update_progress=update_progress,
                enable_weather=enable_weather,
                day_filter=day_filter
            )
        elif self.capabilities.month_filtering_available and month_filter:
            from core import main_with_month_filter
            return main_with_month_filter(
                temp_file_paths, temp_dir,
                update_progress=update_progress,
                enable_weather=enable_weather,
                month_filter=month_filter
            )
        else:
            from core import main as admissions_main
            return admissions_main(
                temp_file_paths, temp_dir,
                update_progress=update_progress,
                enable_weather=enable_weather,
                year_filter=year_range
            )

    def _generate_filename(self, base_name: str, config: Dict[str, Any], extension: str) -> str:
        """Generate appropriate filename based on filter configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if config.get('day_filter'):
            safe_desc = config['day_filter']['description'].replace(' ', '_').replace('-', '_').replace(':',
                                                                                                        '').replace(',',
                                                                                                                    '')
            return f"{base_name}_day_{safe_desc}_{timestamp}.{extension}"
        elif config.get('month_filter'):
            safe_desc = config['month_filter']['description'].replace(' ', '_').replace('-', '_').replace(':',
                                                                                                          '').replace(
                ',', '')
            return f"{base_name}_month_{safe_desc}_{timestamp}.{extension}"
        elif config.get('year_range'):
            return f"{base_name}_year_{config['year_range'][0]}_{config['year_range'][1]}_{timestamp}.{extension}"
        else:
            return f"{base_name}_all_{timestamp}.{extension}"


# ════════════════════════════════════════════════════════════════════════════════
# 6. ENHANCED PERSISTENT RESULTS DISPLAY WITH BETTER ERROR HANDLING
# ════════════════════════════════════════════════════════════════════════════════

class PersistentResultsDisplay:
    """Enhanced results display with better error handling and user experience"""

    @staticmethod
    def show_results_header():
        """Enhanced results header with better session management"""
        if not SessionStateManager.has_valid_results():
            st.warning("No valid results available or session has expired")
            return False

        remaining_time = SessionStateManager.get_time_remaining()
        minutes_remaining = int(remaining_time // 60)
        seconds_remaining = int(remaining_time % 60)

        # Show success message with metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.success("Processing Complete!")

        with col2:
            if remaining_time > 0:
                st.info(f"Time Remaining: {minutes_remaining}m {seconds_remaining}s")
            else:
                st.warning("Session Expired")
                return False

        with col3:
            if st.session_state.processing_duration is not None:
                st.info(f"Processing Time: {st.session_state.processing_duration:.2f}m")

        # Show any error messages from processing
        if st.session_state.error_messages:
            with st.expander("Processing Warnings/Errors", expanded=False):
                for error in st.session_state.error_messages:
                    st.warning(f"⚠️ {error}")

        return True

    @staticmethod
    def show_persistent_downloads():
        """Enhanced download interface with validation"""
        if not SessionStateManager.has_valid_results():
            st.warning("Results are no longer available. Please process new files.")
            if st.button("Start Over", type="primary"):
                SessionStateManager.clear_results()
                st.rerun()
            return

        # Show results header
        if not PersistentResultsDisplay.show_results_header():
            return

        st.divider()

        # Create enhanced download sections
        download_col1, download_col2 = st.columns(2)

        # EU Visits Download
        with download_col1:
            if st.session_state.eu_visits_data is not None:
                PersistentResultsDisplay._show_enhanced_eu_visits_download()
            else:
                st.info("EU Visits: Not processed in this session")

        # Admissions Download
        with download_col2:
            if st.session_state.admissions_data is not None:
                PersistentResultsDisplay._show_enhanced_admissions_download()
            else:
                st.info("Admissions: Not processed in this session")

        # Show enhanced processing summary
        PersistentResultsDisplay._show_enhanced_processing_summary()

        # Show visualizations section
        PersistentResultsDisplay._show_visualizations_section()

        # Show action buttons
        PersistentResultsDisplay._show_action_buttons()

    @staticmethod
    def _show_enhanced_eu_visits_download():
        """FIXED: Enhanced EU Visits download with safer date parsing"""
        st.subheader("EU Visits Data")

        eu_data = st.session_state.eu_visits_data
        filename = st.session_state.eu_visits_filename or "eu_visits_data.csv"

        # Validate data
        if eu_data is None or eu_data.empty:
            st.error("EU Visits data is empty or corrupted")
            return

        # Show metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Records", f"{len(eu_data):,}")
        with col2:
            if 'Visit_Count' in eu_data.columns:
                total_visits = eu_data['Visit_Count'].sum()
                st.metric("Total Visits", f"{total_visits:,}")
        with col3:
            st.metric("Columns", f"{len(eu_data.columns)}")

        # Enhanced data preview with FIXED date parsing
        with st.expander(f"Data Preview ({len(eu_data):,} records)", expanded=False):
            try:
                # FIXED: Use safer date parsing
                if 'Date_String' in eu_data.columns and len(eu_data) > 0:
                    parsed_dates, format_used = safe_date_parsing(eu_data, 'Date_String')

                    if parsed_dates is not None:
                        valid_dates = parsed_dates.dropna()
                        if not valid_dates.empty:
                            first_date = valid_dates.min().strftime('%d/%m/%Y')
                            last_date = valid_dates.max().strftime('%d/%m/%Y')
                            st.success(f"Date Range: {first_date} to {last_date}")
                            if format_used != "automatic":
                                st.info(f"Date format detected: {format_used}")
                    else:
                        st.warning("Could not parse date column - dates may be in unexpected format")

                # Show sample data
                st.dataframe(eu_data.head(20), use_container_width=True)

                # Show data quality information
                missing_data = eu_data.isnull().sum()
                if missing_data.sum() > 0:
                    st.warning("Some columns contain missing data:")
                    missing_cols = missing_data[missing_data > 0]
                    for col, count in missing_cols.items():
                        st.write(f"• {col}: {count} missing values")

            except Exception as e:
                st.error(f"Error in data preview: {str(e)}")

        # FIXED: Use memory-efficient download
        try:
            with st.spinner("Preparing download..."):
                csv_data = memory_efficient_csv_download(eu_data)

            st.download_button(
                "Download EU Visits Data",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                help="Download EU visits data (CSV format)",
                key=f"eu_download_{st.session_state.session_id}",
                use_container_width=True
            )

            # Show file size info
            csv_size_mb = len(csv_data.encode('utf-8')) / (1024 * 1024)
            st.info(f"Download size: {csv_size_mb:.1f} MB")

        except Exception as e:
            st.error(f"Error preparing download: {str(e)}")

    @staticmethod
    def _show_enhanced_admissions_download():
        """Enhanced Admissions download with EXACT SAME features as EU visits"""
        st.subheader("Admissions Data")

        admissions_data = st.session_state.admissions_data
        filename = st.session_state.admissions_filename or "admissions_data.xlsx"

        # Validate data - SAME AS EU VISITS
        if admissions_data is None or len(admissions_data) == 0:
            st.error("Admissions data is empty or corrupted")
            return

        # Read Excel data for full analysis - SAME APPROACH AS EU VISITS
        try:
            admissions_bytes = io.BytesIO(admissions_data)
            df_data = pd.read_excel(admissions_bytes, sheet_name=None)

            # Handle multiple sheets like EU visits handles single dataframe
            if isinstance(df_data, dict):
                # Get main dataframe (first sheet or combine all)
                main_df = list(df_data.values())[0] if df_data else pd.DataFrame()
                total_records = sum(len(df) for df in df_data.values())
                sheet_names = list(df_data.keys())
            else:
                main_df = df_data
                total_records = len(df_data)
                sheet_names = ["Sheet1"]

        except Exception as e:
            st.error(f"Error reading admissions data: {str(e)}")
            return

        # Show metrics - EXACT SAME AS EU VISITS with additions
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Records", f"{total_records:,}")  # SAME AS EU VISITS
        with col2:
            # SAME LOGIC AS EU VISITS - look for visit/admission count column
            count_columns = [col for col in main_df.columns if
                             any(word in col.lower() for word in ['count', 'visits', 'admissions', 'total'])]
            if count_columns:
                count_col = count_columns[0]
                total_count = main_df[count_col].sum() if pd.api.types.is_numeric_dtype(main_df[count_col]) else len(
                    main_df)
                st.metric(f"Total {count_col.replace('_', ' ').title()}", f"{total_count:,}")
            else:
                # Show file size as fallback
                file_size_mb = len(admissions_data) / (1024 * 1024)
                st.metric("File Size", f"{file_size_mb:.1f} MB")
        with col3:
            st.metric("Columns", f"{len(main_df.columns)}")  # SAME AS EU VISITS

        # Enhanced data preview - EXACT SAME AS EU VISITS
        with st.expander(f"Data Preview ({total_records:,} records)", expanded=False):
            try:
                # EXACT SAME date parsing logic as EU Visits
                date_columns = [col for col in main_df.columns if
                                any(word in col.lower() for word in ['date', 'time']) or
                                'Date_String' in main_df.columns]

                # Try Date_String first (same as EU visits)
                if 'Date_String' in main_df.columns and len(main_df) > 0:
                    parsed_dates, format_used = safe_date_parsing(main_df, 'Date_String')

                    if parsed_dates is not None:
                        valid_dates = parsed_dates.dropna()
                        if not valid_dates.empty:
                            first_date = valid_dates.min().strftime('%d/%m/%Y')
                            last_date = valid_dates.max().strftime('%d/%m/%Y')
                            st.success(f"Date Range: {first_date} to {last_date}")
                            if format_used != "automatic":
                                st.info(f"Date format detected: {format_used}")
                    else:
                        st.warning("Could not parse date column - dates may be in unexpected format")
                elif date_columns:
                    # Try other date columns
                    for date_col in date_columns:
                        try:
                            parsed_dates, format_used = safe_date_parsing(main_df, date_col)
                            if parsed_dates is not None:
                                valid_dates = parsed_dates.dropna()
                                if not valid_dates.empty:
                                    first_date = valid_dates.min().strftime('%d/%m/%Y')
                                    last_date = valid_dates.max().strftime('%d/%m/%Y')
                                    st.success(f"Date Range ({date_col}): {first_date} to {last_date}")
                                    if format_used != "automatic":
                                        st.info(f"Date format detected: {format_used}")
                                    break
                        except Exception:
                            continue
                    else:
                        st.warning("Could not parse date columns - dates may be in unexpected format")

                # Show sample data - EXACT SAME AS EU VISITS
                st.dataframe(main_df.head(20), use_container_width=True)

                # Show multiple sheets info if applicable
                if len(sheet_names) > 1:
                    st.info(f"Excel file contains {len(sheet_names)} sheets: {', '.join(sheet_names)}")

                # Show data quality information - EXACT SAME AS EU VISITS
                missing_data = main_df.isnull().sum()
                if missing_data.sum() > 0:
                    st.warning("Some columns contain missing data:")
                    missing_cols = missing_data[missing_data > 0]
                    for col, count in missing_cols.items():
                        st.write(f"• {col}: {count} missing values")

            except Exception as e:
                st.error(f"Error in data preview: {str(e)}")

        # ENHANCED Download with memory efficiency like EU visits
        try:
            with st.spinner("Preparing download..."):
                # Keep original Excel format but show preparation like EU visits
                download_data = admissions_data

            st.download_button(
                "Download Admissions Data",  # Keep admissions-specific text
                data=download_data,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Download admissions data (Excel format)",  # Same help style as EU visits
                key=f"admissions_download_{st.session_state.session_id}",  # Same key pattern
                use_container_width=True  # SAME AS EU VISITS
            )

            # Show file size info - SAME FORMAT AS EU VISITS
            file_size_mb = len(admissions_data) / (1024 * 1024)
            st.info(f"Download size: {file_size_mb:.1f} MB")  # EXACT SAME FORMAT

        except Exception as e:
            st.error(f"Error preparing download: {str(e)}")  # SAME ERROR FORMAT

    @staticmethod
    def _show_enhanced_processing_summary():
        """Enhanced processing summary with detailed configuration"""
        st.divider()
        st.subheader("Processing Summary")

        config = st.session_state.processing_config
        files_info = st.session_state.uploaded_files_info

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            process_type = config.get('process_option', 'Unknown')
            st.metric("Process Type", process_type)

        with col2:
            files_count = len(files_info) if files_info else 0
            st.metric("Files Processed", files_count)

        with col3:
            if st.session_state.processing_duration:
                duration_text = f"{st.session_state.processing_duration:.1f}m"
            else:
                duration_text = "N/A"
            st.metric("Processing Time", duration_text)

        with col4:
            features_count = sum([
                1 if config.get('enable_weather') else 0,
                1 if config.get('enable_bank_holidays') else 0
            ])
            st.metric("Enhanced Features", features_count)

        # Detailed configuration
        with st.expander("Configuration Details", expanded=False):
            # Filter information
            if config.get('day_filter'):
                st.write(f"**Filter:** Day Range - {config['day_filter']['description']}")
                st.write(f"**Duration:** {config['day_filter']['duration_days']} days")
            elif config.get('month_filter'):
                st.write(f"**Filter:** Month Range - {config['month_filter']['description']}")
            elif config.get('year_range'):
                st.write(f"**Filter:** Year Range - {config['year_range'][0]} to {config['year_range'][1]}")
            else:
                st.write("**Filter:** None (All Data)")

            # Enhanced features
            enhanced_features = []
            if config.get('enable_weather'):
                enhanced_features.append("Weather Data")
            if config.get('enable_bank_holidays'):
                enhanced_features.append(f"Bank Holidays ({config.get('uk_region', 'england-and-wales')})")

            if enhanced_features:
                st.write(f"**Enhanced Features:** {', '.join(enhanced_features)}")

            # Files processed
            if files_info:
                st.write("**Files Processed:**")
                for i, file_info in enumerate(files_info, 1):
                    st.write(f"  {i}. {file_info}")

            # Processing timestamp
            if st.session_state.processing_timestamp:
                timestamp_str = st.session_state.processing_timestamp.strftime('%Y-%m-%d %H:%M:%S')
                st.write(f"**Processed:** {timestamp_str}")

    @staticmethod
    def _show_visualizations_section():
        """Enhanced visualizations section with better error handling"""
        st.divider()
        st.subheader("Analytics & Visualizations")

        # Check if we have data for visualizations
        has_eu_data = st.session_state.eu_visits_data is not None
        has_admissions_data = st.session_state.admissions_data is not None
        has_any_data = has_eu_data or has_admissions_data

        if not has_any_data:
            st.info("Analytics Dashboard available after data processing")
            return

        # Check visualization module availability
        viz_available = st.session_state.get('viz_module_available', False)

        if viz_available:
            # Show analytics options
            col1, col2 = st.columns([2, 1])

            with col1:
                st.success("Professional Healthcare Analytics Dashboard Available")

                # Show different descriptions based on available data
                if has_eu_data and has_admissions_data:
                    st.markdown("""

                    """)
                elif has_eu_data:
                    st.markdown("""

                    """)
                elif has_admissions_data:
                    st.markdown("""

                    """)

            with col2:
                pass  # Dataset overview removed

            # Analytics button
            if not st.session_state.get('viz_loading', False):
                button_text = "Generate Analytics Dashboard"
                help_text = "Generate comprehensive healthcare analytics dashboard"

                if st.button(button_text, type="primary", use_container_width=True, help=help_text):
                    st.session_state.viz_loading = True
                    st.session_state.show_visualizations = True
                    st.rerun()
            else:
                with st.spinner("Loading analytics dashboard..."):
                    st.info("Preparing visualizations - this may take a moment...")
                    time.sleep(1)
                    st.session_state.viz_loading = False
                    st.rerun()

            # Show analytics if requested
            if st.session_state.get('show_visualizations', False):
                PersistentResultsDisplay._render_analytics_dashboard()

        else:
            # Show setup instructions
            st.warning("Analytics Dashboard requires visualizations.py module")
            with st.expander("Setup Instructions", expanded=False):
                st.markdown("""
                **To enable the analytics dashboard:**

                1. **Install Required Packages:**
                ```bash
                pip install plotly pandas numpy
                ```

                2. **Add Visualization Module:**
                - Ensure `visualizations.py` is in the same directory as this app
                - The module contains professional healthcare visualizations

                3. **Restart Application:**
                - Restart the Streamlit app after adding the module

                4. **Generate Analytics:**
                - Process your data again to enable analytics
                """)

    @staticmethod
    def _get_data_summary():
        """FIXED: Get summary of available data - removed redundant logic and improved error handling"""
        summary = {}

        try:
            has_eu_data = st.session_state.eu_visits_data is not None
            has_admissions_data = st.session_state.admissions_data is not None

            # Process EU data if available
            if has_eu_data:
                try:
                    eu_data = st.session_state.eu_visits_data
                    eu_records = len(eu_data)

                    summary["EU Records"] = f"{eu_records:,}"

                    # Get visit count if available
                    if 'Visit_Count' in eu_data.columns:
                        try:
                            total_visits = eu_data['Visit_Count'].sum()
                            if pd.notna(total_visits):
                                summary["EU Visits"] = f"{total_visits:,}"
                        except Exception:
                            pass

                    # FIXED: Use safer date parsing
                    if 'Date_String' in eu_data.columns:
                        try:
                            parsed_dates, _ = safe_date_parsing(eu_data, 'Date_String')
                            if parsed_dates is not None:
                                valid_dates = parsed_dates.dropna()
                                if not valid_dates.empty:
                                    date_range = f"{valid_dates.min().strftime('%b %Y')} - {valid_dates.max().strftime('%b %Y')}"
                                    summary["EU Period"] = date_range
                        except Exception:
                            pass

                    # Check for enhanced features in EU data
                    eu_features = []
                    try:
                        if 'Weather_Category' in eu_data.columns and eu_data['Weather_Category'].notna().any():
                            eu_features.append("Weather")
                        if 'Is_Bank_Holiday' in eu_data.columns:
                            eu_features.append("Holidays")
                        if eu_features:
                            summary["EU Features"] = ", ".join(eu_features)
                    except Exception:
                        pass

                except Exception as e:
                    summary["EU Status"] = f"Error: {str(e)[:30]}..."

            # Process Admissions data if available
            if has_admissions_data:
                try:
                    file_size_mb = len(st.session_state.admissions_data) / (1024 * 1024)
                    summary["Admissions Size"] = f"{file_size_mb:.1f} MB"

                    # Try to read Excel data ONCE
                    admissions_bytes = io.BytesIO(st.session_state.admissions_data)
                    df_admissions = pd.read_excel(admissions_bytes)

                    if not df_admissions.empty:
                        admissions_records = len(df_admissions)
                        summary["Admissions Records"] = f"{admissions_records:,}"
                        summary["Admissions Columns"] = f"{len(df_admissions.columns)}"

                        # FIXED: Use safer date parsing for admissions
                        date_columns = [col for col in df_admissions.columns if
                                        any(word in col.lower() for word in ['date', 'time', 'admission', 'discharge'])]

                        for date_col in date_columns:
                            try:
                                parsed_dates, _ = safe_date_parsing(df_admissions, date_col)
                                if parsed_dates is not None:
                                    valid_dates = parsed_dates.dropna()
                                    if not valid_dates.empty:
                                        date_range = f"{valid_dates.min().strftime('%b %Y')} - {valid_dates.max().strftime('%b %Y')}"
                                        summary["Admissions Period"] = date_range
                                        break
                            except Exception:
                                continue

                        # Check for enhanced features in admissions data
                        admissions_features = []
                        try:
                            weather_columns = [col for col in df_admissions.columns if 'weather' in col.lower()]
                            holiday_columns = [col for col in df_admissions.columns if
                                               any(word in col.lower() for word in ['holiday', 'bank'])]

                            if weather_columns:
                                admissions_features.append("Weather")
                            if holiday_columns:
                                admissions_features.append("Holidays")
                            if admissions_features:
                                summary["Admissions Features"] = ", ".join(admissions_features)
                        except Exception:
                            pass
                    else:
                        summary["Admissions Status"] = "Empty file"

                except Exception as e:
                    summary["Admissions Status"] = f"Excel file ready (read error: {str(e)[:20]}...)"

            # FIXED: Calculate combined totals if both datasets exist
            if has_eu_data and has_admissions_data:
                try:
                    eu_count = len(st.session_state.eu_visits_data)

                    # Only read admissions count if we haven't already
                    if "Admissions Records" in summary:
                        # Extract number from existing summary
                        admissions_str = summary["Admissions Records"].replace(",", "")
                        admissions_count = int(admissions_str)
                    else:
                        try:
                            admissions_bytes = io.BytesIO(st.session_state.admissions_data)
                            df_admissions = pd.read_excel(admissions_bytes)
                            admissions_count = len(df_admissions)
                        except Exception:
                            admissions_count = 0

                    if admissions_count > 0:
                        total_records = eu_count + admissions_count
                        summary["Total Records"] = f"{total_records:,}"

                except Exception:
                    pass

            # Add overall status if no data
            if not has_eu_data and not has_admissions_data:
                summary["Status"] = "No data available"

        except Exception as e:
            summary["Status"] = f"Error loading summary: {str(e)[:50]}..."

        return summary  # Helper function that should be defined elsewhere in your code

    @staticmethod
    def _render_analytics_dashboard():
        """Render analytics dashboard with enhanced error handling"""
        st.markdown("---")

        # Header with hide option
        col1, col2 = st.columns([3, 1])
        with col1:
            st.header("Healthcare Analytics Dashboard")
            st.caption("Professional insights across multiple strategic areas")
        with col2:
            if st.button("Hide Analytics", key="hide_analytics"):
                st.session_state.show_visualizations = False
                st.session_state.viz_loading = False
                st.rerun()

        # Render visualizations
        try:
            with st.spinner("Generating comprehensive analytics dashboard..."):
                from visualizations import HealthcareAnalyzer

                # Initialize and run analyzer
                visualizer = HealthcareAnalyzer()
                visualizer.show_dashboard()

                st.session_state.viz_data_processed = True
                st.success("Analytics dashboard generated successfully!")

        except ImportError as e:
            UIHelpers.show_error_message(
                "Visualization Module Missing",
                "Please ensure visualizations.py is in the same directory as this app",
                f"ImportError: {str(e)}"
            )
        except Exception as e:
            UIHelpers.show_error_message(
                "Error Loading Analytics Dashboard",
                "An error occurred while generating the analytics dashboard",
                traceback.format_exc()
            )

        # Action buttons
        st.divider()
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Refresh Analytics", key="refresh_analytics"):
                st.session_state.viz_data_processed = False
                st.session_state.viz_loading = True
                st.rerun()

        with col2:
            if st.button("Hide Analytics", key="hide_analytics_bottom"):
                st.session_state.show_visualizations = False
                st.session_state.viz_loading = False
                st.rerun()

        with col3:
            if st.button("Quick Stats", key="quick_stats"):
                PersistentResultsDisplay._show_quick_stats()

    @staticmethod
    def _show_quick_stats():
        """Show quick statistics modal"""
        with st.expander("Quick Statistics", expanded=True):
            if st.session_state.eu_visits_data is not None:
                data = st.session_state.eu_visits_data

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Records", f"{len(data):,}")

                with col2:
                    if 'Visit_Count' in data.columns:
                        total_visits = data['Visit_Count'].sum()
                        st.metric("Total Visits", f"{total_visits:,}")

                with col3:
                    if 'Date' in data.columns:
                        unique_dates = data['Date'].nunique()
                        st.metric("Days Covered", f"{unique_dates:,}")

                with col4:
                    if 'Outcome_Type' in data.columns:
                        unique_outcomes = data['Outcome_Type'].nunique()
                        st.metric("Categories", f"{unique_outcomes}")

    @staticmethod
    def _show_action_buttons():
        """Show enhanced action buttons"""
        st.divider()
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Process New Files", type="primary", use_container_width=True):
                SessionStateManager.clear_results()
                st.rerun()

        with col2:
            if st.button("Refresh Session", use_container_width=True):
                st.rerun()

        with col3:
            remaining_time = SessionStateManager.get_time_remaining()
            if remaining_time > 0:
                minutes = int(remaining_time // 60)
                seconds = int(remaining_time % 60)
                st.info(f"Session: {minutes}m {seconds}s")
            else:
                st.error("Session Expired")


# ════════════════════════════════════════════════════════════════════════════════
# 7. ENHANCED MAIN APPLICATION CLASS
# ════════════════════════════════════════════════════════════════════════════════

class HealthcareDataProcessorApp:
    """Enhanced main application class with better error handling and user experience"""

    def __init__(self):
        # Initialize session state
        SessionStateManager.initialize_session_state()

        # Initialize components with error handling
        try:
            self.capabilities = ModuleCapabilities()
            self.filters = FilterComponents(self.capabilities)
            self.processor = ProcessingManager(self.capabilities)
        except Exception as e:
            st.error(f"Failed to initialize application components: {str(e)}")
            st.stop()

    def run(self):
        """Enhanced main application entry point"""
        try:
            self._show_header()

            # Check for valid persistent results
            if SessionStateManager.has_valid_results():
                self._show_persistent_results_page()
            else:
                # Clear expired results
                if st.session_state.processing_complete and SessionStateManager.is_session_expired():
                    SessionStateManager.clear_results()

                # Show processing interface
                self._show_processing_interface()

        except Exception as e:
            UIHelpers.show_error_message(
                "Application Error",
                f"An unexpected error occurred: {str(e)}",
                traceback.format_exc()
            )

    def _show_header(self):
        """Enhanced header with module status"""
        st.title("Reduced Demand Of Emergency Unit")
        

       
        st.divider()

    def _show_persistent_results_page(self):
        """Enhanced persistent results page"""
        PersistentResultsDisplay.show_persistent_downloads()
        self._show_enhanced_sidebar()

    def _show_enhanced_sidebar(self):
        """Enhanced sidebar with better information display"""
        with st.sidebar:
            st.subheader("Current Session")

            # Session status
            remaining_time = SessionStateManager.get_time_remaining()
            minutes_remaining = int(remaining_time // 60)
            seconds_remaining = int(remaining_time % 60)

            if remaining_time > 0:
                st.success(f"Active: {minutes_remaining}m {seconds_remaining}s remaining")
            else:
                st.error("Session Expired")

            # Processing info
            config = st.session_state.processing_config
            st.divider()
            st.write("**Session Details:**")

            process_option = config.get('process_option', 'Unknown')
            st.write(f"• **Type:** {process_option}")

            # Filter info
            if config.get('day_filter'):
                st.write(f"• **Filter:** Day Range")
            elif config.get('month_filter'):
                st.write(f"• **Filter:** Month Range")
            elif config.get('year_range'):
                st.write(f"• **Filter:** Year Range ({config['year_range'][0]}-{config['year_range'][1]})")
            else:
                st.write("• **Filter:** None")

            # Enhanced features
            enhanced_features = []
            if config.get('enable_weather'):
                enhanced_features.append("Weather")
            if config.get('enable_bank_holidays'):
                enhanced_features.append("Holidays")

            if enhanced_features:
                st.write(f"• **Enhanced:** {', '.join(enhanced_features)}")

            # Processing metrics
            if st.session_state.processing_duration:
                st.write(f"• **Duration:** {st.session_state.processing_duration:.1f}m")

            # Analytics status
            if st.session_state.eu_visits_data is not None:
                st.divider()
                st.write("**Analytics Status:**")

                viz_available = st.session_state.get('viz_module_available', False)
                if viz_available:
                    st.success("Dashboard Available")

                    if st.session_state.get('show_visualizations', False):
                        st.info("Currently Displayed")
                        if st.button("Hide Dashboard", use_container_width=True):
                            st.session_state.show_visualizations = False
                            st.rerun()
                    else:
                        if st.button("Show Dashboard", use_container_width=True):
                            st.session_state.show_visualizations = True
                            st.rerun()
                else:
                    st.warning("Module Required")

            # Quick actions
            st.divider()
            st.write("**Quick Actions:**")

            if st.button("New Processing", type="primary", use_container_width=True):
                SessionStateManager.clear_results()
                st.rerun()

            if remaining_time > 0:
                if st.button("Refresh Timer", use_container_width=True):
                    st.rerun()

    def _show_processing_interface(self):
        """Enhanced processing interface with better validation"""
        # Show welcome message for new users
        if st.session_state.get('first_visit', True):
            self._show_welcome_message()
            st.session_state.first_visit = False

        # Show available modules
        processing_options = self.capabilities.get_available_processing_options()
        if "No modules available" in processing_options:
            st.error("No processing modules are available. Please check your installation.")
            return

        # Main interface layout
        col1, col2 = st.columns([1, 2])

        with col1:
            filter_config = self._show_configuration_panel()

        with col2:
            self._show_processing_panel(filter_config)

    def _show_welcome_message(self):
        """Enhanced welcome message"""
        st.info("""
        **Welcome to the Healthcare Data Processor & Analytics Platform!**

        """)

    def _show_configuration_panel(self) -> Dict[str, Any]:
        """Fixed configuration panel with mandatory enhanced features"""
        st.subheader("Processing Configuration")

        # Processing options based on available modules
        processing_options = self.capabilities.get_available_processing_options()

        if len(processing_options) == 1 and processing_options[0] != "No modules available":
            process_option = processing_options[0]
            st.info(f"Only {process_option} module is available")
        else:
            process_option = st.radio(
                "Choose what to process:",
                processing_options,
                index=min(2, len(processing_options) - 1),
                help="Select which type of healthcare data to process"
            )

        # Show analytics availability
        if process_option != "No modules available":
            if st.session_state.get('viz_module_available', False):
                st.success("Analytics Dashboard: Available")
            else:
                st.info("Analytics Dashboard: Requires visualizations.py")

        # Date filtering section
        st.subheader("Date Filtering")
        filter_options = self.filters.get_filter_options()
        filter_type = st.radio(
            "Choose filtering precision:",
            filter_options,
            index=0,
            help="Select your date filtering precision level"
        )

        # Initialize filter variables
        year_range = None
        month_filter = None
        day_filter = None

        # Process filter selection with validation
        if filter_type == "No Filter (All Data)":
            st.info("All available data will be included")
        elif filter_type == "Year Range":
            year_range = self.filters.show_year_range_filter()
        elif filter_type == "Month Range" and self.capabilities.month_filtering_available:
            month_filter = self.filters.show_month_range_filter()
        elif filter_type == "Day Range" and self.capabilities.day_filtering_available:
            day_filter = self.filters.show_day_range_filter()

        # Enhanced features - NOW MANDATORY AND NON-INTERACTIVE
        st.subheader("Enhanced Features")

        # Option 1: Simple success messages (most straightforward)
        st.success("Weather data included - Enables seasonal analysis")
        st.success("Bank holidays included - Enables holiday impact analysis")

        # Option 2: Info boxes with explanations (alternative - comment out Option 1 to use)
        # st.info("Weather Data: Always included for comprehensive seasonal analysis")
        # st.info("Bank Holidays: Always included for holiday impact analysis")

        # Option 3: Metric-style display (alternative - comment out Option 1 to use)
        # col1, col2 = st.columns(2)
        # with col1:
        #     st.metric("Weather Data", "Enabled", "Always included")
        # with col2:
        #     st.metric("Bank Holidays", "Enabled", "Always included")

        # UK Region selection (still interactive since it's a configuration choice)
        uk_region = st.selectbox(
            "UK Region for Bank Holidays:",
            options=["england-and-wales", "scotland", "northern-ireland"],
            index=0,
            help="Select appropriate UK region for bank holiday calendar"
        )

        # Always return True for mandatory features
        return {
            'process_option': process_option,
            'year_range': year_range,
            'month_filter': month_filter,
            'day_filter': day_filter,
            'enable_weather': True,  # Always True - mandatory
            'enable_bank_holidays': True,  # Always True - mandatory
            'uk_region': uk_region
        }

    def _show_processing_panel(self, config: Dict[str, Any]):
        """Enhanced processing panel with better validation and user guidance"""
        st.subheader("File Upload & Processing")

        # File upload with enhanced validation
        uploaded_files = st.file_uploader(
            "Choose Excel files",
            type=['xlsx', 'xls'],
            accept_multiple_files=True,
            help="Upload one or more Excel files containing healthcare data"
        )

        if uploaded_files:
            # Validate files
            files_valid, validation_errors = UIHelpers.validate_uploaded_files(uploaded_files)

            if not files_valid:
                st.error("File validation failed:")
                for error in validation_errors:
                    st.error(f"• {error}")
                return

            # Show file details
            total_size = sum(file.size for file in uploaded_files)
            total_size_mb = total_size / (1024 * 1024)

            

            # Show file list
            with st.expander("File Details", expanded=False):
                for i, file in enumerate(uploaded_files, 1):
                    size_mb = file.size / (1024 * 1024)
                    st.write(f"{i}. **{file.name}** ({size_mb:.1f} MB)")

            # Show configuration summary
            self._show_enhanced_config_summary(config)

            # Show analytics preview
            if config['process_option'] != "No modules available":
                self._show_analytics_preview(config)

            # Process button with enhanced validation
            if self._validate_processing_config(config):
                if st.button("Process Data", type="primary", use_container_width=True):
                    self._process_data_with_enhanced_handling(uploaded_files, config)
            else:
                st.warning("Please correct configuration issues before processing")

        else:
            # Show enhanced usage instructions
            st.info("Please upload Excel files to begin processing")
            self._show_enhanced_usage_guide()

    def _show_enhanced_config_summary(self, config: Dict[str, Any]):
        """Updated config summary for mandatory features"""
        with st.expander("Processing Configuration Summary", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Processing Settings:**")
                st.write(f"• **Type:** {config['process_option']}")

                # Filter summary
                if config['day_filter']:
                    st.write(f"• **Filter:** Day Range ({config['day_filter']['duration_days']} days)")
                elif config['month_filter']:
                    st.write(f"• **Filter:** Month Range")
                elif config['year_range']:
                    st.write(f"• **Filter:** Year Range ({config['year_range'][0]}-{config['year_range'][1]})")
                else:
                    st.write("• **Filter:** None (All Data)")

            with col2:
                st.write("**Enhanced Features:**")

                # Always show as enabled since they're mandatory
                st.write("• **Weather Data:** Always Enabled")
                st.write(f"• **Bank Holidays:** Always Enabled ({config['uk_region']})")

                # Analytics status
                if st.session_state.get('viz_module_available', False):
                    st.write("• **Analytics Dashboard:** Available")
                else:
                    st.write("• **Analytics Dashboard:** Requires Module")

    def _show_analytics_preview(self, config: Dict[str, Any]):
        """Updated analytics preview for mandatory features"""
        if st.session_state.get('viz_module_available', False):
            if config['process_option'] == "Only EU Visits":
                st.success("EU Visits Analytics Dashboard will be generated with clinical and operational insights")
            elif config['process_option'] == "Only Admissions":
                st.success("Admissions Analytics Dashboard will be generated with hospital performance insights")
            elif config['process_option'] == "Both":
                st.success("Comprehensive Analytics Dashboard will be generated with integrated insights")
        else:
            st.info("Analytics dashboard will be available after adding visualizations.py module")

    def _validate_processing_config(self, config: Dict[str, Any]) -> bool:
        """Validate processing configuration"""
        if config['process_option'] == "No modules available":
            st.error("No processing modules available")
            return False

        # Validate filters
        if config.get('day_filter') is None and config.get('month_filter') is None and config.get('year_range') is None:
            # No filter is fine
            pass

        return True

    def _show_enhanced_usage_guide(self):
        """Updated usage guide reflecting mandatory features"""
        with st.expander("Complete User Guide", expanded=True):
            tab1, tab2 = st.tabs(["Quick Start", "Troubleshooting"])

            with tab1:
                st.markdown("""
                **Quick Start Guide:**

                1. **Upload Files** - Click the file uploader and select Excel files (.xlsx or .xls)
                2. **Choose Processing Type** - Select EU Visits, Admissions, or Both
                3. **Set Date Filter** - Choose year range, month range, or day range (optional)
                4. **Select UK Region** - Choose appropriate region for bank holidays
                5. **Process Data** - Click "Process Data" (weather and holidays automatically included)
                6. **Download Results** - Use download buttons (results persist for 15+ minutes)
                7. **View Analytics** - Generate professional dashboard for insights
                """)

           

            with tab2:
                st.markdown("""
                **Troubleshooting:**

                **Common Issues:**
                • **File Upload Errors:** Ensure files are .xlsx or .xls format, under 100MB
                • **Processing Failures:** Check file content and try smaller batches
                • **Weather Data Issues:** System automatically falls back without weather if needed
                • **Analytics Not Available:** Ensure visualizations.py module is present

                **Performance Tips:**
                • **Large Files:** Process in smaller batches for better performance
                • **Network Issues:** Weather/holiday data will automatically fallback if unreachable
                • **Memory Issues:** Close other applications if processing large datasets

                **Module Requirements:**
                • **EU_Visits.py:** Required for EU visits processing
                • **core.py:** Required for admissions processing
                • **visualizations.py:** Required for analytics dashboard (optional)
                """)

    def _process_data_with_enhanced_handling(self, uploaded_files, config: Dict[str, Any]):
        """Enhanced data processing with better progress tracking and error handling"""
        # Create progress containers
        progress_container = st.container()
        status_container = st.container()

        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()

        temp_dir = None
        temp_file_paths = []

        try:
            # Show processing start message
            with status_container:
                st.info("Starting comprehensive data processing...")
                if config['process_option'] in ["Only EU Visits", "Both"] and st.session_state.get(
                        'viz_module_available', False):
                    st.info("Analytics dashboard will be available after processing")

            # Record start time
            st.session_state.processing_start_time = datetime.now()

            # Save files temporarily
            status_text.text("Saving files temporarily...")
            progress_bar.progress(10)

            temp_dir = tempfile.mkdtemp(prefix="healthcare_proc_")
            for uploaded_file in uploaded_files:
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                temp_file_paths.append(temp_path)

            progress_bar.progress(25)
            status_text.text("Processing healthcare data...")
            progress_bar.progress(50)

            # Process data with enhanced error handling
            success = self.processor.process_and_save_results(
                temp_file_paths, temp_dir, config, uploaded_files
            )

            if success:
                progress_bar.progress(90)
                status_text.text("Preparing results...")
                time.sleep(1)

                progress_bar.progress(100)
                status_text.text("Processing complete!")

                with status_container:
                
                    st.success("Processing Complete! Your healthcare data is ready.")
                    st.info("Session Active: Results available for 15+ minutes")

                    if config['process_option'] in ["Only EU Visits", "Both"] and st.session_state.get(
                            'viz_module_available', False):
                        st.success("Analytics Ready: Professional dashboard with insights now available!")

                # Brief pause then redirect
                time.sleep(2)
                st.rerun()
            else:
                with status_container:
                    st.error("No data was processed successfully")
                    st.info("Please check your files and configuration, then try again")

        except Exception as e:
            with status_container:
                UIHelpers.show_error_message(
                    "Processing Failed",
                    f"Critical error during processing: {str(e)}",
                    traceback.format_exc()
                )

                # Show troubleshooting help
                st.markdown("""
                **Troubleshooting Steps:**
                1. **Check Files:** Ensure Excel files contain valid healthcare data
                2. **File Format:** Use .xlsx or .xls format only  
                3. **File Size:** Very large files may timeout - try smaller batches
                4. **Dependencies:** Ensure all required modules are available
                5. **Restart:** Try restarting the Streamlit application
                """)

        finally:
            # Cleanup temp files
            try:
                if temp_dir and os.path.isdir(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass  # Silent cleanup failure


# ════════════════════════════════════════════════════════════════════════════════
# 8. ENHANCED UTILITY FUNCTIONS AND APPLICATION ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════════

def check_system_requirements():
    """Enhanced system requirements checking"""
    requirements_met = True
    analytics_available = True
    missing_packages = []

    # Check core requirements
    required_packages = ['pandas', 'streamlit']
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            requirements_met = False
            missing_packages.append(package)

    # Check optional requirements for analytics
    try:
        import plotly
    except ImportError:
        analytics_available = False

    return requirements_met, analytics_available, missing_packages


def run_main_app():
    """Run the main application with comprehensive error handling"""
    try:
        # Check system requirements first
        requirements_met, analytics_available, missing_packages = check_system_requirements()

        if not requirements_met:
            st.error("Missing Required Dependencies")
            for package in missing_packages:
                st.error(f"• Missing: {package}")

            st.markdown("""
            **Installation Instructions:**
            ```bash
            pip install streamlit pandas plotly
            ```

            Please install the missing packages and restart the application.
            """)
            st.stop()

        # Show startup status
        with st.spinner("Initializing Healthcare Data Processor..."):
            app = HealthcareDataProcessorApp()

        # Run the application
        app.run()

    except ImportError as e:
        st.error("Missing Dependencies")
        st.markdown(f"""
        **Required module missing:** `{str(e)}`

        **Installation Steps:**
        1. Install required packages:
        ```bash
        pip install streamlit pandas plotly
        ```

        2. Ensure core modules are available:
        - `EU_Visits.py` - for EU visits processing
        - `core.py` - for admissions processing  
        - `visualizations.py` - for analytics dashboard (optional)

        3. Restart the application
        """)

    except Exception as e:
        st.error("Application Startup Failed")
        st.error(f"Error: {str(e)}")

        with st.expander("Startup Error Details"):
            st.code(traceback.format_exc())

        st.markdown("""
        **Troubleshooting:**
        - **Dependencies:** Ensure all required Python packages are installed
        - **Modules:** Check that EU_Visits.py and core.py are available
        - **Environment:** Verify your Python environment is properly configured
        - **Restart:** Try restarting the Streamlit application
        - **Support:** Check console output for additional error messages
        """)

        # Show system information for debugging
        with st.expander("System Information"):
            import sys
            import platform

            st.write(f"**Python Version:** {sys.version}")
            st.write(f"**Platform:** {platform.platform()}")
            st.write(f"**Streamlit Version:** {st.__version__}")

            # Check for dependencies
            st.write("**Dependencies Status:**")
            try:
                import plotly
                st.write(f"• Plotly: Available ({plotly.__version__})")
            except ImportError:
                st.write("• Plotly: Not installed")

            try:
                import pandas as pd
                st.write(f"• Pandas: Available ({pd.__version__})")
            except ImportError:
                st.write("• Pandas: Not installed")


def main():
    """Application entry point with proper initialization"""
    # Check system requirements
    requirements_met, analytics_available, missing_packages = check_system_requirements()

    if not requirements_met:
        st.error("System Requirements Not Met")
        for package in missing_packages:
            st.error(f"Missing: {package}")
        st.stop()

    # Run the main application
    run_main_app()


# ════════════════════════════════════════════════════════════════════════════════
# 9. APPLICATION ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()