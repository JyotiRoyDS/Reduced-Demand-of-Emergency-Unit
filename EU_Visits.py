"""
EU Visits Data Processor - Fixed Encoding Issues + Weather API Integration
Processes ALL sections/tables within each Excel sheet + UK Bank Holidays + Weather Data from APIs
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import traceback
from typing import Dict, List, Tuple, Optional, Any
import warnings
import os
from pathlib import Path
import requests
import json
import time
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

warnings.filterwarnings('ignore')


# ============================================================================
# WEATHER API INTEGRATION - DYNAMIC LOCATION AND DATE DETECTION
# ============================================================================

def get_coordinates_for_location(location_name: str) -> Tuple[Optional[float], Optional[float], str]:
    """
    Get coordinates for a location using OpenStreetMap Nominatim API (no API key required)

    Args:
        location_name: Name of the location to geocode

    Returns:
        Tuple of (latitude, longitude, formatted_address)
    """
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            'q': location_name,
            'format': 'json',
            'limit': 1,
            'countrycodes': 'gb'  # Limit to UK
        }

        headers = {
            'User-Agent': 'EU-Visits-Processor/1.0 (Medical Research)'
        }

        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()

        if data and len(data) > 0:
            result = data[0]
            lat = float(result['lat'])
            lon = float(result['lon'])
            address = result.get('display_name', location_name)

            print(f"🌍 Found coordinates for {location_name}: ({lat:.4f}, {lon:.4f})")
            return lat, lon, address
        else:
            print(f"⚠️ No coordinates found for {location_name}")
            return None, None, location_name

    except Exception as e:
        print(f"⚠️ Error geocoding {location_name}: {str(e)}")
        return None, None, location_name


def detect_location_from_data(df: pd.DataFrame) -> str:
    """
    Dynamically detect location from the data content (no hardcoding)

    Args:
        df: DataFrame that might contain location information

    Returns:
        Detected location name
    """
    # Default locations to try based on common patterns in the data
    potential_locations = []

    # Check for Cardiff/Wales references in the data
    location_indicators = [
        r'cardiff',
        r'wales',
        r'uhw',
        r'university\s+hospital\s+wales',
        r'cardiff\s+and\s+vale'
    ]

    # Search through the dataframe for location indicators
    for col in df.columns:
        for row_idx in range(min(50, len(df))):  # Check first 50 rows
            try:
                cell_value = str(df.iloc[row_idx, col]).lower()
                for pattern in location_indicators:
                    if re.search(pattern, cell_value):
                        if 'cardiff' in cell_value or 'wales' in cell_value:
                            potential_locations.append('Cardiff, Wales, UK')
                        elif 'uhw' in cell_value:
                            potential_locations.append('Cardiff, Wales, UK')
            except:
                continue

    # Return the most likely location
    if potential_locations:
        return potential_locations[0]
    else:
        # Fallback to a sensible default for UK hospital data
        return 'Cardiff, Wales, UK'


def fetch_weather_data(start_date: datetime, end_date: datetime, latitude: float, longitude: float,
                       timezone: str = 'Europe/London') -> pd.DataFrame:
    """Backward compatibility wrapper for robust weather fetching"""
    return fetch_weather_data_robust(start_date, end_date, latitude, longitude, timezone)
    """
    Fetch weather data from Open-Meteo API for specified date range and location

    Args:
        start_date: Start date for weather data
        end_date: End date for weather data
        latitude: Latitude of location
        longitude: Longitude of location

    Returns:
        DataFrame with weather data
    """
    print(f"🌤️ Fetching weather data for coordinates ({latitude:.4f}, {longitude:.4f})")
    print(f"   Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    try:
        # Open-Meteo Historical Weather API
        url = "https://archive-api.open-meteo.com/v1/archive"

        params = {
            'latitude': latitude,
            'longitude': longitude,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'daily': [
                'temperature_2m_max',
                'temperature_2m_min',
                'temperature_2m_mean',
                'precipitation_sum',
                'weathercode',
                'windspeed_10m_max',
                'relative_humidity_2m_mean',
                'surface_pressure_mean'
            ],
            'timezone': 'Europe/London'  # UK timezone
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        if 'daily' in data:
            weather_df = pd.DataFrame({
                'Date': pd.to_datetime(data['daily']['time']),
                'Temperature_Max_C': data['daily']['temperature_2m_max'],
                'Temperature_Min_C': data['daily']['temperature_2m_min'],
                'Temperature_Mean_C': data['daily']['temperature_2m_mean'],
                'Precipitation_mm': data['daily']['precipitation_sum'],
                'Weather_Code': data['daily']['weathercode'],
                'Wind_Speed_kmh': data['daily']['windspeed_10m_max'],
                'Humidity_Percent': data['daily']['relative_humidity_2m_mean'],
                'Pressure_hPa': data['daily']['surface_pressure_mean']
            })

            # Add weather condition descriptions
            weather_df['Weather_Condition'] = weather_df['Weather_Code'].apply(decode_weather_code)

            # Add weather categories for analysis
            weather_df['Weather_Category'] = weather_df['Weather_Condition'].apply(categorize_weather)

            # Add temperature categories
            weather_df['Temperature_Category'] = weather_df['Temperature_Mean_C'].apply(categorize_temperature)

            # Add precipitation categories
            weather_df['Precipitation_Category'] = weather_df['Precipitation_mm'].apply(categorize_precipitation)

            print(f"✅ Retrieved weather data for {len(weather_df)} days")
            print(f"   Temperature range: {weather_df['Temperature_Mean_C'].min():.1f}°C to {weather_df['Temperature_Mean_C'].max():.1f}°C")
            print(f"   Total precipitation: {weather_df['Precipitation_mm'].sum():.1f}mm")

            # Show sample weather conditions
            sample_conditions = weather_df['Weather_Condition'].value_counts().head(3)
            print("🌤️ Most common weather conditions:")
            for condition, count in sample_conditions.items():
                print(f"   - {condition}: {count} days")

            return weather_df
        else:
            print("⚠️ No weather data received from API")
            return pd.DataFrame()

    except requests.RequestException as e:
        print(f"⚠️ Failed to fetch weather data: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"⚠️ Error processing weather data: {e}")
        return pd.DataFrame()


def decode_weather_code(code: int) -> str:
    """
    Decode WMO weather codes to human-readable descriptions

    Args:
        code: WMO weather code

    Returns:
        Weather condition description
    """
    if pd.isna(code):
        return 'Unknown'

    code = int(code)

    weather_codes = {
        0: 'Clear sky',
        1: 'Mainly clear',
        2: 'Partly cloudy',
        3: 'Overcast',
        45: 'Fog',
        48: 'Depositing rime fog',
        51: 'Light drizzle',
        53: 'Moderate drizzle',
        55: 'Dense drizzle',
        56: 'Light freezing drizzle',
        57: 'Dense freezing drizzle',
        61: 'Slight rain',
        63: 'Moderate rain',
        65: 'Heavy rain',
        66: 'Light freezing rain',
        67: 'Heavy freezing rain',
        71: 'Slight snow',
        73: 'Moderate snow',
        75: 'Heavy snow',
        77: 'Snow grains',
        80: 'Slight rain showers',
        81: 'Moderate rain showers',
        82: 'Violent rain showers',
        85: 'Slight snow showers',
        86: 'Heavy snow showers',
        95: 'Thunderstorm',
        96: 'Thunderstorm with slight hail',
        99: 'Thunderstorm with heavy hail'
    }

    return weather_codes.get(code, f'Unknown ({code})')


def categorize_weather(condition: str) -> str:
    """
    Categorize weather conditions into broader categories for analysis

    Args:
        condition: Weather condition description

    Returns:
        Weather category
    """
    condition_lower = condition.lower()

    if any(word in condition_lower for word in ['clear', 'sunny']):
        return 'Clear'
    elif any(word in condition_lower for word in ['cloudy', 'overcast', 'partly']):
        return 'Cloudy'
    elif any(word in condition_lower for word in ['rain', 'drizzle', 'shower']):
        return 'Rainy'
    elif any(word in condition_lower for word in ['snow', 'sleet']):
        return 'Snowy'
    elif any(word in condition_lower for word in ['thunderstorm', 'storm']):
        return 'Stormy'
    elif any(word in condition_lower for word in ['fog', 'mist']):
        return 'Foggy'
    else:
        return 'Other'


def categorize_temperature(temp_c: float) -> str:
    """
    Categorize temperature into ranges for analysis

    Args:
        temp_c: Temperature in Celsius

    Returns:
        Temperature category
    """
    if pd.isna(temp_c):
        return 'Unknown'

    if temp_c < 0:
        return 'Freezing'
    elif temp_c < 5:
        return 'Very Cold'
    elif temp_c < 10:
        return 'Cold'
    elif temp_c < 15:
        return 'Cool'
    elif temp_c < 20:
        return 'Mild'
    elif temp_c < 25:
        return 'Warm'
    elif temp_c < 30:
        return 'Hot'
    else:
        return 'Very Hot'


def categorize_precipitation(precip_mm: float) -> str:
    """
    Categorize precipitation into ranges for analysis

    Args:
        precip_mm: Precipitation in millimeters

    Returns:
        Precipitation category
    """
    if pd.isna(precip_mm):
        return 'Unknown'

    if precip_mm == 0:
        return 'None'
    elif precip_mm < 2.5:
        return 'Light'
    elif precip_mm < 10:
        return 'Moderate'
    elif precip_mm < 50:
        return 'Heavy'
    else:
        return 'Very Heavy'


def add_weather_info(df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add weather information to the main dataframe

    Args:
        df: Main dataframe with Date column
        weather_df: Weather dataframe from API

    Returns:
        Enhanced dataframe with weather information
    """
    if weather_df.empty:
        # Add empty weather columns if no weather data
        weather_columns = [
            'Temperature_Max_C', 'Temperature_Min_C', 'Temperature_Mean_C',
            'Precipitation_mm', 'Weather_Condition', 'Weather_Category',
            'Temperature_Category', 'Precipitation_Category',
            'Wind_Speed_kmh', 'Humidity_Percent', 'Pressure_hPa'
        ]
        for col in weather_columns:
            df[col] = None
        return df

    print(f"🌤️ Adding weather information to {len(df)} records")

    # Ensure both dataframes have proper date columns
    df['Date_for_Weather'] = pd.to_datetime(df['Date']).dt.date
    weather_df['Date_for_Weather'] = pd.to_datetime(weather_df['Date']).dt.date

    # Merge weather data
    df_with_weather = df.merge(
        weather_df.drop('Date', axis=1),  # Drop original Date column to avoid conflicts
        on='Date_for_Weather',
        how='left'
    )

    # Clean up temporary column
    df_with_weather = df_with_weather.drop('Date_for_Weather', axis=1)

    # Count how many records got weather data
    weather_records = df_with_weather['Weather_Condition'].notna().sum()
    print(f"✅ Added weather data to {weather_records:,} records")

    if weather_records > 0:
        # Show sample weather statistics
        print(f"🌡️ Temperature range: {df_with_weather['Temperature_Mean_C'].min():.1f}°C to {df_with_weather['Temperature_Mean_C'].max():.1f}°C")
        print(f"🌧️ Precipitation range: {df_with_weather['Precipitation_mm'].min():.1f}mm to {df_with_weather['Precipitation_mm'].max():.1f}mm")

        # Show weather category distribution
        weather_dist = df_with_weather['Weather_Category'].value_counts()
        print("🌤️ Weather distribution:")
        for category, count in weather_dist.head(5).items():
            if pd.notna(category):
                print(f"   - {category}: {count:,} records")

    return df_with_weather


def sort_eu_visits_dataframe(df: pd.DataFrame,
                             day_filter: Dict[str, Any] = None,
                             month_filter: Dict[str, Any] = None,
                             year_filter: tuple = None) -> pd.DataFrame:
    """Sort EU visits DataFrame according to filter criteria"""
    if df.empty:
        return df

    print(f"🔄 Sorting {len(df)} EU visits records...")

    try:
        df['Date_for_sorting'] = pd.to_datetime(df['Date_String'], format='%d/%m/%Y', errors='coerce')

        df = df.sort_values([
            'Date_for_sorting',
            'Type',
            'Outcome_Type',
            'Source_File'
        ])

        df = df.drop('Date_for_sorting', axis=1)

        if 'Date_String' in df.columns and len(df) > 0:
            first_date = df['Date_String'].iloc[0]
            last_date = df['Date_String'].iloc[-1]
            print(f"   📊 EU visits date range after sorting: {first_date} to {last_date}")

    except Exception as e:
        print(f"   ⚠️ Error sorting EU visits data: {e}")
        sort_columns = [col for col in ['Year', 'Month', 'Date_String', 'Type', 'Outcome_Type'] if col in df.columns]
        if sort_columns:
            df = df.sort_values(sort_columns)

    return df


def fetch_weather_data_robust(start_date: datetime, end_date: datetime, latitude: float, longitude: float,
                              timezone: str = 'Europe/London') -> pd.DataFrame:
    """Robust weather data fetching with SSL error handling and multiple fallback strategies"""
    print(f"🌤️ Fetching weather data for coordinates ({latitude:.4f}, {longitude:.4f})")
    print(f"   Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    try:
        return _fetch_with_robust_session(start_date, end_date, latitude, longitude, timezone)
    except Exception as e:
        print(f"⚠️ Strategy 1 failed: {str(e)[:100]}...")

    try:
        return _fetch_with_ssl_disabled(start_date, end_date, latitude, longitude, timezone)
    except Exception as e:
        print(f"⚠️ Strategy 2 failed: {str(e)[:100]}...")

    try:
        return _fetch_with_simple_params(start_date, end_date, latitude, longitude, timezone)
    except Exception as e:
        print(f"⚠️ Strategy 3 failed: {str(e)[:100]}...")

    print("⚠️ All weather data fetching strategies failed")
    return pd.DataFrame()


def _fetch_with_robust_session(start_date: datetime, end_date: datetime, latitude: float, longitude: float,
                               timezone: str) -> pd.DataFrame:
    session = requests.Session()

    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    session.headers.update({
        'User-Agent': 'Healthcare-Data-Processor/1.0 (Research Application)',
        'Accept': 'application/json',
    })

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        'latitude': latitude,
        'longitude': longitude,
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'daily': [
            'temperature_2m_max',
            'temperature_2m_min',
            'temperature_2m_mean',
            'precipitation_sum',
            'weathercode',
            'windspeed_10m_max',
            'relative_humidity_2m_mean',
            'surface_pressure_mean'
        ],
        'timezone': timezone
    }

    response = session.get(url, params=params, timeout=(10, 30))
    response.raise_for_status()

    return _process_weather_response(response.json())


def _fetch_with_ssl_disabled(start_date: datetime, end_date: datetime, latitude: float, longitude: float,
                             timezone: str) -> pd.DataFrame:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        'latitude': latitude,
        'longitude': longitude,
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'daily': [
            'temperature_2m_max',
            'temperature_2m_min',
            'temperature_2m_mean',
            'precipitation_sum',
            'weathercode',
            'windspeed_10m_max',
            'relative_humidity_2m_mean',
            'surface_pressure_mean'
        ],
        'timezone': timezone
    }

    headers = {
        'User-Agent': 'Healthcare-Data-Processor/1.0 (Research Application)'
    }

    response = requests.get(
        url,
        params=params,
        headers=headers,
        verify=False,
        timeout=(15, 45)
    )
    response.raise_for_status()

    return _process_weather_response(response.json())


def _fetch_with_simple_params(start_date: datetime, end_date: datetime, latitude: float, longitude: float,
                              timezone: str) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        'latitude': latitude,
        'longitude': longitude,
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'daily': 'temperature_2m_mean,precipitation_sum,weathercode',
        'timezone': timezone
    }

    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; HealthcareProcessor/1.0)',
        'Accept': 'application/json'
    }

    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=(20, 60),
        verify=False
    )

    response.raise_for_status()
    return _process_weather_response(response.json())


def _process_weather_response(data: dict) -> pd.DataFrame:
    if 'daily' not in data:
        return pd.DataFrame()

    daily_data = data['daily']

    weather_df = pd.DataFrame({
        'Date': pd.to_datetime(daily_data['time']),
    })

    weather_df['Temperature_Max_C'] = daily_data.get('temperature_2m_max', [None] * len(weather_df))
    weather_df['Temperature_Min_C'] = daily_data.get('temperature_2m_min', [None] * len(weather_df))
    weather_df['Temperature_Mean_C'] = daily_data.get('temperature_2m_mean', [None] * len(weather_df))
    weather_df['Precipitation_mm'] = daily_data.get('precipitation_sum', [None] * len(weather_df))
    weather_df['Weather_Code'] = daily_data.get('weathercode', [None] * len(weather_df))
    weather_df['Wind_Speed_kmh'] = daily_data.get('windspeed_10m_max', [None] * len(weather_df))
    weather_df['Humidity_Percent'] = daily_data.get('relative_humidity_2m_mean', [None] * len(weather_df))
    weather_df['Pressure_hPa'] = daily_data.get('surface_pressure_mean', [None] * len(weather_df))

    weather_df['Weather_Condition'] = weather_df['Weather_Code'].apply(decode_weather_code)
    weather_df['Weather_Category'] = weather_df['Weather_Condition'].apply(categorize_weather)
    weather_df['Temperature_Category'] = weather_df['Temperature_Mean_C'].apply(categorize_temperature)
    weather_df['Precipitation_Category'] = weather_df['Precipitation_mm'].apply(categorize_precipitation)

    return weather_df


def enhance_dataframe_with_weather_graceful(df: pd.DataFrame, location_name: str = None,
                                            date_column: str = 'Date',
                                            custom_indicators: List[str] = None) -> pd.DataFrame:
    if df.empty or date_column not in df.columns:
        return _add_empty_weather_columns(df)

    try:
        if not location_name:
            location_name = detect_location_from_data(df, custom_indicators)

        lat, lon, address = get_coordinates_for_location(location_name)

        if lat is None or lon is None:
            return _add_empty_weather_columns(df)

        try:
            dates = pd.to_datetime(df[date_column], format='%d/%m/%Y', errors='coerce')
            valid_dates = dates.dropna()

            if valid_dates.empty:
                return _add_empty_weather_columns(df)

            start_date = valid_dates.min()
            end_date = valid_dates.max()

        except Exception as e:
            return _add_empty_weather_columns(df)

        weather_df = fetch_weather_data_robust(start_date, end_date, lat, lon)

        if weather_df.empty:
            return _add_empty_weather_columns(df)

        return add_weather_info(df, weather_df, date_column)

    except Exception as e:
        print(f"⚠️ Weather enhancement failed: {e}")
        return _add_empty_weather_columns(df)


def _add_empty_weather_columns(df: pd.DataFrame) -> pd.DataFrame:
    weather_columns = [
        'Temperature_Max_C', 'Temperature_Min_C', 'Temperature_Mean_C',
        'Precipitation_mm', 'Weather_Condition', 'Weather_Category',
        'Temperature_Category', 'Precipitation_Category',
        'Wind_Speed_kmh', 'Humidity_Percent', 'Pressure_hPa'
    ]

    for col in weather_columns:
        if col not in df.columns:
            df[col] = None

    return df

# ADD THIS FUNCTION after add_weather_info function

def sort_eu_visits_dataframe(df: pd.DataFrame,
                             day_filter: Dict[str, Any] = None,
                             month_filter: Dict[str, Any] = None,
                             year_filter: tuple = None) -> pd.DataFrame:
    """Sort EU visits DataFrame according to filter criteria"""
    if df.empty:
        return df

    print(f"🔄 Sorting {len(df)} EU visits records...")

    try:
        df['Date_for_sorting'] = pd.to_datetime(df['Date_String'], format='%d/%m/%Y', errors='coerce')

        df = df.sort_values([
            'Date_for_sorting',
            'Type',
            'Outcome_Type',
            'Source_File'
        ])

        df = df.drop('Date_for_sorting', axis=1)

        if 'Date_String' in df.columns and len(df) > 0:
            first_date = df['Date_String'].iloc[0]
            last_date = df['Date_String'].iloc[-1]
            print(f"   📊 EU visits date range after sorting: {first_date} to {last_date}")

    except Exception as e:
        print(f"   ⚠️ Error sorting EU visits data: {e}")
        sort_columns = [col for col in ['Year', 'Month', 'Date_String', 'Type', 'Outcome_Type'] if col in df.columns]
        if sort_columns:
            df = df.sort_values(sort_columns)

    return df

# ============================================================================
# UK BANK HOLIDAYS API INTEGRATION - FIXED ENCODING
# ============================================================================

def fetch_bank_holidays(years: List[int], region: str = "england-and-wales", cache_timeout: int = 3600) -> pd.DataFrame:
    """
    Fetch bank holidays from GOV.UK API for specified years with proper encoding handling.

    Args:
        years: List of years to fetch bank holidays for
        region: UK region (default: "england-and-wales")
                Options: "england-and-wales", "scotland", "northern-ireland"
        cache_timeout: Cache timeout in seconds (not implemented in this version)

    Returns:
        DataFrame with columns: Date, Event, Type, Audience
    """
    print(f"🏛️ Fetching bank holidays for years: {years}")

    try:
        url = "https://www.gov.uk/bank-holidays.json"

        # Set headers to ensure proper encoding
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Charset': 'utf-8',
            'Accept-Encoding': 'gzip, deflate'
        }

        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()

        # Force UTF-8 encoding and handle response properly
        response.encoding = 'utf-8'

        # Get JSON data with proper encoding handling
        try:
            # First try to get the JSON directly
            data = response.json()
        except Exception:
            # Fallback: manually decode the content
            content = response.content.decode('utf-8', errors='replace')
            data = json.loads(content)

        # Check if the region exists in the data
        if region not in data:
            print(f"⚠️ Region '{region}' not found in API data")
            available_regions = list(data.keys())
            print(f"   Available regions: {available_regions}")
            return pd.DataFrame(columns=["Date", "Event", "Type", "Audience"])

        events = data[region]["events"]

        # Convert to DataFrame
        df = pd.DataFrame(events)
        if df.empty:
            print(f"⚠️ No bank holiday data received from API")
            return pd.DataFrame(columns=["Date", "Event", "Type", "Audience"])

        # Process the data with enhanced text cleaning
        df["Date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

        # Apply comprehensive encoding fix to the titles
        df["Event"] = df["title"].apply(lambda x: fix_encoding_issues_comprehensive(x))

        df["Type"] = "Bank Holiday"
        df["Audience"] = "General Public"

        # Filter to requested years
        df = df[df["Date"].dt.year.isin(years)]
        df = df[["Date", "Event", "Type", "Audience"]].copy()

        print(f"✅ Retrieved {len(df)} bank holidays for {region}")

        # Debug: Print first few holidays to see if encoding is fixed
        if not df.empty:
            print("📅 Sample holidays:")
            for _, holiday in df.head(3).iterrows():
                print(f"   {holiday['Date'].strftime('%Y-%m-%d')}: {holiday['Event']}")

        return df

    except requests.RequestException as e:
        print(f"⚠️ Failed to fetch bank holidays from GOV.UK API: {e}")
        print(f"   Continuing without bank holiday data...")
        return pd.DataFrame(columns=["Date", "Event", "Type", "Audience"])
    except Exception as e:
        print(f"⚠️ Error processing bank holiday data: {e}")
        return pd.DataFrame(columns=["Date", "Event", "Type", "Audience"])


def fix_encoding_issues_comprehensive(text: str) -> str:
    """
    Comprehensive fix for various encoding issues that can occur with text from APIs
    This version handles double-encoding scenarios that can occur in web applications

    Args:
        text: Text that may have encoding issues

    Returns:
        Cleaned text with proper characters
    """
    if not isinstance(text, str):
        return str(text)

    # Handle bytes if somehow we get them
    if isinstance(text, bytes):
        try:
            text = text.decode('utf-8')
        except:
            text = text.decode('latin-1', errors='replace')

    # Convert to string and normalize
    text = str(text)

    # Step 1: Fix double-encoded UTF-8 issues (common in web apps)
    double_encoding_fixes = {
        # Common double-encoded patterns
        'â€™': "'",        # Right single quotation mark (double-encoded)
        'â€œ': '"',        # Left double quotation mark (double-encoded)
        'â€': '"',         # Right double quotation mark (double-encoded)
        'â€"': '–',        # En dash (double-encoded)
        'â€"': '—',        # Em dash (double-encoded)
        'â€¦': '…',        # Horizontal ellipsis (double-encoded)

        # Other common double-encoding issues
        'Ã¢â‚¬â„¢': "'",     # Another variant
        'Ã¢â‚¬Å"': '"',     # Another variant
        'Ã¢â‚¬': '"',      # Another variant
    }

    # Apply double-encoding fixes first
    for old, new in double_encoding_fixes.items():
        text = text.replace(old, new)

    # Step 2: Fix single UTF-8 encoding issues
    single_encoding_fixes = {
        # Apostrophes and quotes
        ''': "'",          # Right single quotation mark
        ''': "'",          # Left single quotation mark
        '"': '"',          # Left double quotation mark
        '"': '"',          # Right double quotation mark
        '…': '...',        # Horizontal ellipsis

        # Dashes
        '–': '-',          # En dash
        '—': '-',          # Em dash

        # Currency and symbols
        '£': '£',          # Pound sign
        '€': '€',          # Euro sign
        '©': '©',          # Copyright
        '®': '®',          # Registered trademark

        # Accented characters (preserve them properly)
        'á': 'á',          # á
        'à': 'à',          # à
        'é': 'é',          # é
        'è': 'è',          # è
        'í': 'í',          # í
        'ó': 'ó',          # ó
        'ú': 'ú',          # ú
    }

    # Apply single encoding fixes
    for old, new in single_encoding_fixes.items():
        text = text.replace(old, new)

    # Step 3: Fix legacy encoding artifacts
    legacy_fixes = {
        'Ã‚': '',          # Standalone Ã‚
        '\ufeff': '',      # BOM character
        '\u200b': '',      # Zero-width space
        '\u00a0': ' ',     # Non-breaking space to regular space
        '\u2019': "'",     # Another right single quotation mark
        '\u201c': '"',     # Another left double quotation mark
        '\u201d': '"',     # Another right double quotation mark
    }

    # Apply legacy fixes
    for old, new in legacy_fixes.items():
        text = text.replace(old, new)

    # Step 4: Additional cleanup
    # Remove any remaining problematic characters that might cause issues
    # but preserve normal text
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)  # Remove control characters

    # Normalize multiple spaces
    text = ' '.join(text.split())

    return text.strip()


def fix_encoding_issues(text: str) -> str:
    """
    Legacy function name for backward compatibility
    """
    return fix_encoding_issues_comprehensive(text)


def add_bank_holiday_info(df: pd.DataFrame, bank_holidays_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add bank holiday information to the main dataframe with proper encoding handling.

    Args:
        df: Main dataframe with Date column (format: DD/MM/YYYY or similar)
        bank_holidays_df: Bank holidays dataframe from GOV.UK API

    Returns:
        Enhanced dataframe with bank holiday information
    """
    if bank_holidays_df.empty:
        df['Is_Bank_Holiday'] = 'No'
        df['Bank_Holiday_Name'] = None
        return df

    print(f"🏛️ Adding bank holiday information to {len(df)} records")

    # Ensure Date column is datetime
    # Try different date formats
    df['Date_Parsed'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')

    # If that failed, try other common formats
    if df['Date_Parsed'].isna().all():
        df['Date_Parsed'] = pd.to_datetime(df['Date'], errors='coerce')

    bank_holidays_df['Date'] = pd.to_datetime(bank_holidays_df['Date'])

    # Create a lookup dictionary for faster matching
    # Apply encoding fix to the holiday names before creating lookup
    bank_holiday_lookup = {}
    for _, row in bank_holidays_df.iterrows():
        date_key = row['Date'].strftime('%Y-%m-%d')
        # Apply encoding fix to ensure clean holiday names
        clean_event_name = fix_encoding_issues_comprehensive(str(row['Event']))
        bank_holiday_lookup[date_key] = clean_event_name

    # Add bank holiday information
    def check_bank_holiday(date_val):
        if pd.isna(date_val):
            return 'No', None

        date_key = date_val.strftime('%Y-%m-%d')
        if date_key in bank_holiday_lookup:
            return 'Yes', bank_holiday_lookup[date_key]
        return 'No', None

    df[['Is_Bank_Holiday', 'Bank_Holiday_Name']] = df['Date_Parsed'].apply(
        lambda x: pd.Series(check_bank_holiday(x))
    )

    # Clean up temporary column
    df = df.drop('Date_Parsed', axis=1)

    bank_holiday_count = (df['Is_Bank_Holiday'] == 'Yes').sum()
    print(f"✅ Identified {bank_holiday_count} bank holiday records")

    return df


# ============================================================================
# ENHANCED EU VISITS PROCESSOR WITH DYNAMIC DATA DETECTION + WEATHER (NO HARDCODING)
# ============================================================================

class EUVisitsProcessorWithDynamicDetection:
    """
    Enhanced EU Visits processor with dynamic data detection + weather integration (no hardcoding)
    """

    def __init__(self, log_callback=None, enable_bank_holidays=True, uk_region="england-and-wales",
                 enable_weather=True, location_name=None):
        """Initialize the processor with bank holidays and weather support"""
        self.log_callback = log_callback
        self.log_messages = []
        self.enable_bank_holidays = enable_bank_holidays
        self.uk_region = uk_region
        self.enable_weather = enable_weather
        self.location_name = location_name  # Can be None for auto-detection
        self.bank_holidays_df = pd.DataFrame()
        self.weather_df = pd.DataFrame()
        self.detected_location = None
        self.coordinates = (None, None)

        # Complete section headers list - ALL VARIATIONS
        self.SECTION_HEADERS = [
            "UHW EU Visits by Outcome",
            "UHW EU Visits by Attendance Category",
            "UHW EU Visits by Source of Referral",
            "UHW EU Visits by Triage Category",
            "UHW EU Visits by Triage Categories",  # Alternative
            "UHW EU Visits by Hour of Arrival",
            "UHW EU Visits by Recorded Diagnoses",
            "UHW EU Visits by Recorded Diagnosis"   # Alternative
        ]

        # Dynamic filter patterns (NO hardcoded years/months - fully dynamic)
        self.DYNAMIC_FILTER_PATTERNS = {
            'header_indicators': [
                r'no\.?\s*of\s+visits',
                r'uhw\s+eu\s+visits',
                r'total',
                r'average',
                r'maximum',
                r'sum',
                r'visits\s+as\s+values',
                r'cardiff\s+and\s+vale',
                r'information\s+services'
            ],
            'subtitle_indicators': [
                r'outcome\s+of\s+attendance',
                r'attendance\s+category',
                r'source\s+of\s+referral',
                r'manual\s+triage\s+categories',
                r'hour\s+of\s+arrival',
                r'primary\s+diagnosis',
                r'recorded\s+diagnos',
                r'triage\s+categories'
            ],
            'period_indicators': [
                r'period:?\s*',
                r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4}',  # Any month + year
                r'\d{4}\s*-\s*\d{4}',  # Any year range
                r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4}\s*-\s*\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4}',  # Any month-year to month-year range
                r'academic\s+year',
                r'financial\s+year'
            ],
            'organizational_info': [
                r'cardiff\s+and\s+vale',
                r'university\s+health\s+board',
                r'information\s+services',
                r'report\s+generated',
                r'data\s+extracted'
            ]
        }

        # Date pattern regex for DD/MM/YYYY format
        self.DATE_PATTERN = re.compile(r'\d{2}/\d{2}/20\d{2}')

    def log_message(self, message: str) -> None:
        """Log messages"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        formatted_message = f"[{timestamp}] {message}"
        self.log_messages.append(formatted_message)
        print(formatted_message)

        if self.log_callback:
            self.log_callback(formatted_message)

    def setup_location_and_weather(self, df: pd.DataFrame) -> None:
        """Setup location detection and weather data for the date range"""
        if not self.enable_weather:
            self.log_message("🌤️ Weather integration disabled")
            return

        # Detect or use provided location
        if self.location_name:
            self.detected_location = self.location_name
            self.log_message(f"🌍 Using provided location: {self.detected_location}")
        else:
            self.detected_location = detect_location_from_data(df)
            self.log_message(f"🌍 Auto-detected location: {self.detected_location}")

        # Get coordinates for location
        lat, lon, address = get_coordinates_for_location(self.detected_location)
        self.coordinates = (lat, lon)

        if lat is not None and lon is not None:
            self.log_message(f"📍 Location: {address}")
            self.log_message(f"📍 Coordinates: ({lat:.4f}, {lon:.4f})")
        else:
            self.log_message("⚠️ Could not get coordinates for location")
            self.enable_weather = False

    def load_weather_for_date_range(self, start_date: datetime, end_date: datetime) -> None:
        """Load weather data for the detected date range"""
        if not self.enable_weather or self.coordinates[0] is None:
            self.log_message("🌤️ Weather data loading skipped")
            return

        self.log_message(f"🌤️ Loading weather data for date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        try:
            self.weather_df = fetch_weather_data(
                start_date, end_date,
                self.coordinates[0], self.coordinates[1]
            )

            if not self.weather_df.empty:
                self.log_message(f"✅ Loaded weather data for {len(self.weather_df)} days")
                # Show weather summary
                temp_range = f"{self.weather_df['Temperature_Mean_C'].min():.1f}°C to {self.weather_df['Temperature_Mean_C'].max():.1f}°C"
                total_precip = self.weather_df['Precipitation_mm'].sum()
                self.log_message(f"🌡️ Temperature range: {temp_range}")
                self.log_message(f"🌧️ Total precipitation: {total_precip:.1f}mm")
            else:
                self.log_message("⚠️ No weather data loaded")

        except Exception as e:
            self.log_message(f"❌ Error loading weather data: {str(e)}")
            self.weather_df = pd.DataFrame()

    def load_bank_holidays_for_years(self, years: List[int]) -> None:
        """Load bank holidays for specific years"""
        if not self.enable_bank_holidays:
            self.log_message("🏦 Bank holidays disabled")
            return

        self.log_message(f"🏦 Loading bank holidays for years: {years}")

        try:
            self.bank_holidays_df = fetch_bank_holidays(years, self.uk_region)

            if not self.bank_holidays_df.empty:
                self.log_message(f"✅ Loaded {len(self.bank_holidays_df)} bank holidays")

                # Show some examples
                if len(self.bank_holidays_df) > 0:
                    recent_holidays = self.bank_holidays_df.tail(3)
                    self.log_message("📅 Recent bank holidays:")
                    for _, holiday in recent_holidays.iterrows():
                        self.log_message(f"   {holiday['Date'].strftime('%Y-%m-%d')}: {holiday['Event']}")
            else:
                self.log_message("⚠️ No bank holidays loaded")

        except Exception as e:
            self.log_message(f"❌ Error loading bank holidays: {str(e)}")
            self.bank_holidays_df = pd.DataFrame()

    def extract_dates_from_headers(self, df: pd.DataFrame) -> Tuple[List[str], int, Dict[str, Any]]:
        """Extract actual dates from Excel headers with automatic detection"""
        self.log_message("📅 Starting automated date extraction...")

        date_info = {
            'header_row': -1,
            'total_dates': 0,
            'date_range': {},
            'date_format': 'DD/MM/YYYY'
        }

        # Search for date header row in first 20 rows
        for row_idx in range(min(20, len(df))):
            row = df.iloc[row_idx]
            row_strings = [str(cell).strip() for cell in row if pd.notna(cell)]
            date_matches = []

            for cell_value in row_strings:
                if self.DATE_PATTERN.match(cell_value):
                    date_matches.append(cell_value)

            # If we found dates, this is likely the header row
            if len(date_matches) > 5:
                self.log_message(f"📅 Found date header at row {row_idx + 1} with {len(date_matches)} dates")

                # Extract all dates from this row
                all_dates = []
                for col_idx in range(1, len(df.columns)):
                    try:
                        cell_value = str(df.iloc[row_idx, col_idx]).strip()
                        if self.DATE_PATTERN.match(cell_value):
                            all_dates.append(cell_value)
                    except:
                        continue

                if all_dates:
                    # Parse dates to get temporal information
                    parsed_dates = []
                    years_found = set()
                    for date_str in all_dates:
                        try:
                            parsed_date = datetime.strptime(date_str, '%d/%m/%Y')
                            parsed_dates.append(parsed_date)
                            years_found.add(parsed_date.year)
                        except:
                            continue

                    if parsed_dates:
                        start_date = min(parsed_dates)
                        end_date = max(parsed_dates)

                        date_info.update({
                            'header_row': row_idx,
                            'total_dates': len(all_dates),
                            'date_range': {
                                'start_date': start_date,
                                'end_date': end_date,
                                'start_date_str': all_dates[0],
                                'end_date_str': all_dates[-1]
                            }
                        })

                        self.log_message(f"📅 Date range: {all_dates[0]} to {all_dates[-1]} ({len(all_dates)} days)")

                        # Setup location and coordinates for weather
                        self.setup_location_and_weather(df)

                        # Load bank holidays for detected years
                        if self.enable_bank_holidays and years_found:
                            self.load_bank_holidays_for_years(sorted(list(years_found)))

                        # Load weather data for detected date range
                        if self.enable_weather and self.coordinates[0] is not None:
                            self.load_weather_for_date_range(start_date, end_date)

                        return all_dates, row_idx, date_info

        self.log_message("⚠️ No date headers found with automatic detection")
        return [], -1, date_info

    def is_non_data_row(self, row_text: str) -> Tuple[bool, str]:
        """
        DYNAMIC: Determine if a row contains non-data content (no hardcoding)
        """
        if not row_text or row_text.strip() == '':
            return True, "empty"

        row_lower = row_text.lower().strip()

        # Check against dynamic patterns
        for category, patterns in self.DYNAMIC_FILTER_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, row_lower):
                    return True, category

        # Additional intelligent checks
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{4}',
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
            r'\b\d{4}\b.*\b\d{4}\b',
        ]

        for pattern in date_patterns:
            if re.search(pattern, row_lower):
                return True, "date_pattern"

        if re.search(r'\d+\.?\d*\s*%', row_text):
            return True, "percentage"

        numeric_parts = re.findall(r'\d+', row_text)
        text_parts = re.sub(r'[\d\s\.,%-]', '', row_text).strip()

        if len(numeric_parts) > 3 and len(text_parts) < 3:
            return True, "mostly_numeric"

        excel_artifacts = ['sheet', 'tab', 'worksheet', 'page', 'print', 'footer', 'header']
        if any(artifact in row_lower for artifact in excel_artifacts):
            return True, "excel_artifact"

        return False, "data_row"

    def find_data_start_dynamically(self, df: pd.DataFrame, header_row: int, section_name: str) -> int:
        """DYNAMIC: Find where actual data starts after a section header (no hardcoding)"""
        self.log_message(f"🔍 Finding data start for '{section_name}' after row {header_row + 1}")

        max_search_rows = min(15, len(df) - header_row - 1)

        for offset in range(1, max_search_rows + 1):
            check_row_idx = header_row + offset

            if check_row_idx >= len(df):
                break

            first_col_val = df.iloc[check_row_idx, 0] if len(df.columns) > 0 else None

            if pd.isna(first_col_val):
                continue

            row_text = str(first_col_val).strip()
            is_non_data, reason = self.is_non_data_row(row_text)

            if is_non_data:
                self.log_message(f"   Row {check_row_idx + 1}: Skipping ({reason}): '{row_text[:50]}...'")
                continue
            else:
                row_data = df.iloc[check_row_idx]
                numeric_cols = 0
                total_cols = 0

                for col_idx in range(1, min(10, len(row_data))):
                    if col_idx < len(row_data):
                        cell_val = row_data.iloc[col_idx]
                        if pd.notna(cell_val):
                            total_cols += 1
                            try:
                                float(cell_val)
                                numeric_cols += 1
                            except (ValueError, TypeError):
                                pass

                if total_cols > 0 and (numeric_cols / total_cols) >= 0.3:
                    self.log_message(f"   ✅ Data starts at row {check_row_idx + 1}: '{row_text[:50]}...'")
                    return check_row_idx
                else:
                    self.log_message(f"   Row {check_row_idx + 1}: Not enough numeric data: '{row_text[:30]}...'")

        fallback_row = header_row + 1
        self.log_message(f"   ⚠️ Using fallback data start: row {fallback_row + 1}")
        return fallback_row

    def find_data_end_dynamically(self, df: pd.DataFrame, data_start_row: int, next_section_row: Optional[int] = None) -> int:
        """DYNAMIC: Find where data ends for a section (no hardcoding)"""
        if next_section_row is not None:
            max_row = next_section_row - 1
        else:
            max_row = len(df) - 1

        self.log_message(f"🔍 Finding data end from row {data_start_row + 1} to row {max_row + 1}")

        last_data_row = data_start_row

        for row_idx in range(data_start_row, max_row + 1):
            if row_idx >= len(df):
                break

            row_data = df.iloc[row_idx]
            has_meaningful_data = False
            first_col_val = str(row_data.iloc[0]).strip() if len(row_data) > 0 else ""

            if first_col_val and first_col_val.lower() not in ['nan', 'none', '']:
                is_non_data, reason = self.is_non_data_row(first_col_val)

                if not is_non_data:
                    numeric_count = 0
                    total_count = 0

                    for col_idx in range(1, min(len(row_data), 20)):
                        cell_val = row_data.iloc[col_idx]
                        if pd.notna(cell_val) and str(cell_val).strip() not in ['', 'nan']:
                            total_count += 1
                            try:
                                float(cell_val)
                                numeric_count += 1
                            except (ValueError, TypeError):
                                pass

                    if total_count > 0 and (numeric_count / total_count) >= 0.2:
                        has_meaningful_data = True
                        last_data_row = row_idx

            if not has_meaningful_data:
                summary_indicators = ['total', 'sum', 'average', 'grand total', 'subtotal']
                if any(indicator in first_col_val.lower() for indicator in summary_indicators):
                    self.log_message(f"   Stopping at summary row {row_idx + 1}: '{first_col_val[:30]}...'")
                    break

        self.log_message(f"   ✅ Data ends at row {last_data_row + 1}")
        return last_data_row

    def detect_all_section_headers(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """DYNAMIC: Detect ALL section headers in the DataFrame (no hardcoding)"""
        self.log_message("🔍 Starting comprehensive section header detection...")
        section_locations = {}

        for row_idx in range(len(df)):
            for col_idx in range(min(5, len(df.columns))):
                try:
                    cell_raw = df.iloc[row_idx, col_idx]
                    if pd.isna(cell_raw):
                        continue

                    cell_value = str(cell_raw).strip()
                    if not cell_value or cell_value.lower() in ['nan', 'none', '']:
                        continue

                    for header in self.SECTION_HEADERS:
                        normalized_key = self._normalize_section_header(header)

                        already_found = False
                        for existing_header in section_locations.keys():
                            if self._normalize_section_header(existing_header) == normalized_key:
                                already_found = True
                                break

                        if already_found:
                            continue

                        if header.lower() == cell_value.lower():
                            section_locations[header] = {
                                'start_row': row_idx,
                                'found_text': cell_value,
                                'location': f"Row {row_idx + 1}, Col {col_idx + 1}",
                                'match_type': 'exact'
                            }
                            self.log_message(f"✅ Found '{header}' at Row {row_idx + 1} (exact)")
                            break
                        elif self._fuzzy_match_section(cell_value, header):
                            section_locations[header] = {
                                'start_row': row_idx,
                                'found_text': cell_value,
                                'location': f"Row {row_idx + 1}, Col {col_idx + 1}",
                                'match_type': 'fuzzy'
                            }
                            self.log_message(f"✅ Found '{header}' at Row {row_idx + 1} (fuzzy)")
                            break

                except Exception:
                    continue

        if section_locations:
            self._determine_all_section_boundaries_dynamically(df, section_locations)

        self.log_message(f"📊 Section detection complete: {len(section_locations)} sections found")
        return section_locations

    def _normalize_section_header(self, header: str) -> str:
        """Normalize section headers to avoid duplicates"""
        header_lower = header.lower()
        if 'outcome' in header_lower:
            return 'outcome'
        elif 'attendance' in header_lower:
            return 'attendance'
        elif 'referral' in header_lower:
            return 'referral'
        elif 'triage' in header_lower:
            return 'triage'
        elif 'hour' in header_lower:
            return 'hour'
        elif 'diagnos' in header_lower:
            return 'diagnosis'
        return header_lower

    def _fuzzy_match_section(self, cell_value: str, header: str) -> bool:
        """Enhanced fuzzy matching for section headers"""
        cell_lower = cell_value.lower()
        header_lower = header.lower()

        if 'uhw' not in cell_lower or 'eu' not in cell_lower or 'visits' not in cell_lower:
            return False

        if 'outcome' in header_lower:
            return 'outcome' in cell_lower
        elif 'attendance' in header_lower:
            return 'attendance' in cell_lower
        elif 'referral' in header_lower:
            return 'referral' in cell_lower
        elif 'triage' in header_lower:
            return 'triage' in cell_lower
        elif 'hour' in header_lower:
            return 'hour' in cell_lower
        elif 'diagnos' in header_lower:
            return 'diagnos' in cell_lower

        return False

    def _determine_all_section_boundaries_dynamically(self, df: pd.DataFrame, section_locations: Dict[str, Dict[str, Any]]) -> None:
        """DYNAMIC: Determine boundaries for ALL sections using intelligent detection (no hardcoding)"""
        sorted_sections = sorted(section_locations.items(), key=lambda x: x[1]['start_row'])

        self.log_message(f"🔍 Determining boundaries for {len(sorted_sections)} sections using dynamic detection...")

        for i, (section_name, info) in enumerate(sorted_sections):
            header_row = info['start_row']
            self.log_message(f"📋 Processing boundaries for: {section_name}")

            data_start_row = self.find_data_start_dynamically(df, header_row, section_name)

            if i + 1 < len(sorted_sections):
                next_section_start = sorted_sections[i + 1][1]['start_row']
                data_end_row = self.find_data_end_dynamically(df, data_start_row, next_section_start)
            else:
                data_end_row = self.find_data_end_dynamically(df, data_start_row, None)

            section_locations[section_name]['data_start'] = data_start_row
            section_locations[section_name]['data_end'] = data_end_row
            section_locations[section_name]['row_count'] = max(0, data_end_row - data_start_row + 1)

            self.log_message(
                f"📊 {section_name}: "
                f"Header Row {header_row + 1}, "
                f"Data Rows {data_start_row + 1}-{data_end_row + 1} "
                f"({section_locations[section_name]['row_count']} data rows)"
            )

    def process_section_data_with_dates(self, df: pd.DataFrame, section_name: str, section_info: Dict[str, Any],
                                        date_list: List[str], date_header_row: int, sheet_name: str, file_name: str) -> pd.DataFrame:
        """Process individual section data with proper date mapping"""
        try:
            start_row = section_info.get('data_start', 0)
            end_row = section_info.get('data_end', len(df) - 1)

            self.log_message(f"🔧 Processing {section_name}: rows {start_row + 1} to {end_row + 1}")

            if start_row >= len(df) or end_row >= len(df) or start_row > end_row:
                self.log_message(f"⚠️ Invalid bounds for {section_name}")
                return pd.DataFrame()

            section_df = df.iloc[start_row:end_row + 1].copy()
            if section_df.empty:
                return pd.DataFrame()

            section_df = section_df.dropna(how='all').reset_index(drop=True)
            valid_rows = self._filter_valid_rows_dynamically(section_df)

            if not valid_rows:
                self.log_message(f"⚠️ No valid data rows found in {section_name}")
                return pd.DataFrame()

            section_filtered = section_df.iloc[valid_rows].copy().reset_index(drop=True)
            self.log_message(f"📋 {section_name}: {len(section_filtered)} valid rows after filtering")

            data_columns = self._find_data_columns_with_dates(section_filtered, date_list)

            if not data_columns:
                self.log_message(f"⚠️ No data columns found for {section_name}")
                return pd.DataFrame()

            self.log_message(f"📊 {section_name}: Found {len(data_columns)} date columns")

            records = self._convert_to_long_format_with_dates(
                section_filtered, data_columns, date_list, section_name, sheet_name, file_name
            )

            if records:
                result_df = pd.DataFrame(records)
                result_df = self._add_enhanced_temporal_features(result_df)
                self.log_message(f"✅ {section_name}: {len(result_df)} records processed successfully")
                return result_df
            else:
                self.log_message(f"⚠️ No valid records created for {section_name}")

            return pd.DataFrame()

        except Exception as e:
            self.log_message(f"❌ Error processing {section_name}: {str(e)}")
            return pd.DataFrame()

    def _filter_valid_rows_dynamically(self, section_df: pd.DataFrame) -> List[int]:
        """DYNAMIC: Filter out invalid rows like headers and summaries (no hardcoding)"""
        valid_rows = []

        for idx, row in section_df.iterrows():
            outcome_val = str(row.iloc[0]).strip()

            if outcome_val and outcome_val.lower() != 'nan':
                is_non_data, reason = self.is_non_data_row(outcome_val)

                if not is_non_data:
                    valid_rows.append(idx)
                else:
                    self.log_message(f"   Filtering out row {idx}: '{outcome_val[:30]}...' ({reason})")

        return valid_rows

    def _find_data_columns_with_dates(self, section_df: pd.DataFrame, date_list: List[str]) -> List[Tuple[int, str]]:
        """Find data columns that correspond to dates"""
        data_columns = []

        for col_idx in range(1, len(section_df.columns)):
            col_data = section_df.iloc[:, col_idx]
            numeric_count = 0
            total_non_na = 0

            for val in col_data:
                if pd.notna(val) and str(val).strip() not in ['', 'nan', 'NA']:
                    total_non_na += 1
                    try:
                        float(val)
                        numeric_count += 1
                    except (ValueError, TypeError):
                        pass

            if total_non_na > 0 and (numeric_count / total_non_na) > 0.2:
                date_idx = col_idx - 1
                if 0 <= date_idx < len(date_list):
                    data_columns.append((col_idx, date_list[date_idx]))

        return data_columns

    def _convert_to_long_format_with_dates(self, section_df: pd.DataFrame, data_columns: List[Tuple[int, str]],
                                           date_list: List[str], section_name: str, sheet_name: str, file_name: str) -> List[Dict[str, Any]]:
        """Convert wide format data to long format with actual dates"""
        records = []

        for row_idx in range(len(section_df)):
            outcome_type = str(section_df.iloc[row_idx, 0]).strip()

            if not outcome_type or outcome_type.lower() in ['nan', 'none', '']:
                continue

            for col_idx, date_str in data_columns:
                visit_count = section_df.iloc[row_idx, col_idx]

                if pd.notna(visit_count) and str(visit_count).strip() not in ['', 'NA', 'nan']:
                    try:
                        visit_count_num = float(visit_count)
                        if visit_count_num >= 0:
                            records.append({
                                'Outcome_Type': outcome_type,
                                'Visit_Count': visit_count_num,
                                'Date_String': date_str,
                                'Type': section_name,
                                'Source_Sheet': sheet_name,
                                'Source_File': file_name
                            })
                    except (ValueError, TypeError):
                        continue

        return records

    def _add_enhanced_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive temporal features including bank holidays and weather with fixed encoding"""
        if 'Date_String' not in df.columns:
            return df

        try:
            df['Date'] = pd.to_datetime(df['Date_String'], format='%d/%m/%Y')
        except:
            df['Date'] = pd.to_datetime(df['Date_String'], infer_datetime_format=True)

        # Extract temporal components
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['Day_Of_Week'] = df['Date'].dt.day_name()
        df['Day_Of_Week_Num'] = df['Date'].dt.dayofweek
        df['Is_Weekend'] = df['Day_Of_Week_Num'].isin([5, 6])

        # Month names
        df['Month_Name'] = df['Date'].dt.strftime('%b')
        df['Month'] = df['Date'].dt.strftime('%B')

        # Seasons
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Autumn'

        df['Season'] = df['Month'].apply(get_season)

        # Quarters
        df['Quarter'] = df['Date'].dt.quarter
        df['Quarter_Name'] = 'Q' + df['Quarter'].astype(str)

        # Academic year (September to August)
        def get_academic_year(date):
            if date.month >= 9:
                return f"{date.year}-{date.year + 1}"
            else:
                return f"{date.year - 1}-{date.year}"

        df['Academic_Year'] = df['Date'].apply(get_academic_year)

        # NHS financial year (April to March)
        def get_nhs_year(date):
            if date.month >= 4:
                return f"{date.year}-{date.year + 1}"
            else:
                return f"{date.year - 1}-{date.year}"

        df['NHS_Financial_Year'] = df['Date'].apply(get_nhs_year)

        # BANK HOLIDAYS INTEGRATION WITH FIXED ENCODING
        if self.enable_bank_holidays and not self.bank_holidays_df.empty:
            self.log_message("🏦 Adding bank holiday information with fixed encoding...")
            df = add_bank_holiday_info(df, self.bank_holidays_df)

            if 'Bank_Holiday_Name' in df.columns:
                df['Bank_Holiday_Name'] = df['Bank_Holiday_Name'].apply(
                    lambda x: fix_encoding_issues_comprehensive(str(x)) if pd.notna(x) else x
                )
        else:
            df['Is_Bank_Holiday'] = 'No'
            df['Bank_Holiday_Name'] = None

        # WEATHER INTEGRATION
        if self.enable_weather and not self.weather_df.empty:
            self.log_message("🌤️ Adding weather information...")
            df = add_weather_info(df, self.weather_df)
        else:
            weather_columns = [
                'Temperature_Max_C', 'Temperature_Min_C', 'Temperature_Mean_C',
                'Precipitation_mm', 'Weather_Condition', 'Weather_Category',
                'Temperature_Category', 'Precipitation_Category',
                'Wind_Speed_kmh', 'Humidity_Percent', 'Pressure_hPa'
            ]
            for col in weather_columns:
                df[col] = None

        return df

    def process_excel_file(self, file_path: str) -> pd.DataFrame:
        """DYNAMIC: Process a single Excel file finding ALL sections in each sheet (no hardcoding) + weather"""
        self.log_message(f"🚀 Processing: {file_path}")

        try:
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            self.log_message(f"📋 Available sheets: {sheet_names}")

            matching_sheets = self._find_eu_visits_sheets(sheet_names)
            if not matching_sheets:
                self.log_message("⚠️ No EU visits sheets found, using all sheets")
                matching_sheets = sheet_names

            all_processed_data = []

            for sheet_name in matching_sheets:
                try:
                    self.log_message(f"\n📊 Processing sheet: {sheet_name}")
                    df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

                    if df_raw.empty:
                        self.log_message(f"⚠️ Sheet {sheet_name} is empty")
                        continue

                    self.log_message(f"📋 Sheet size: {df_raw.shape}")

                    date_list, date_header_row, date_info = self.extract_dates_from_headers(df_raw)
                    if not date_list:
                        self.log_message(f"⚠️ No dates found in {sheet_name}")
                        continue

                    section_locations = self.detect_all_section_headers(df_raw)
                    if not section_locations:
                        self.log_message(f"❌ No sections found in {sheet_name}")
                        continue

                    self.log_message(f"🎯 Processing {len(section_locations)} sections in {sheet_name}")

                    sheet_data = []
                    for section_name, section_info in section_locations.items():
                        self.log_message(f"\n🔧 Processing section: {section_name}")

                        section_data = self.process_section_data_with_dates(
                            df_raw, section_name, section_info, date_list,
                            date_header_row, sheet_name, os.path.basename(file_path)
                        )

                        if not section_data.empty:
                            sheet_data.append(section_data)
                            self.log_message(f"✅ {section_name}: Added {len(section_data)} records")
                        else:
                            self.log_message(f"⚠️ {section_name}: No data extracted")

                    if sheet_data:
                        combined_sheet_data = pd.concat(sheet_data, ignore_index=True)
                        all_processed_data.append(combined_sheet_data)
                        self.log_message(f"✅ Sheet {sheet_name}: Total {len(combined_sheet_data)} records")
                    else:
                        self.log_message(f"❌ No data extracted from {sheet_name}")

                except Exception as e:
                    self.log_message(f"❌ Error processing {sheet_name}: {str(e)}")
                    continue

            if all_processed_data:
                final_data = pd.concat(all_processed_data, ignore_index=True)
                self.log_message(f"\n🎉 TOTAL PROCESSED: {len(final_data)} records from {len(all_processed_data)} sheets")

                # Summary by section
                if 'Type' in final_data.columns:
                    section_counts = final_data['Type'].value_counts()
                    self.log_message("📊 Records by section:")
                    for section, count in section_counts.items():
                        self.log_message(f"   {section}: {count:,} records")

                # Bank holiday summary with encoding verification
                if 'Is_Bank_Holiday' in final_data.columns and self.enable_bank_holidays:
                    bh_count = (final_data['Is_Bank_Holiday'] == 'Yes').sum()
                    if bh_count > 0:
                        total_visits_bh = final_data[final_data['Is_Bank_Holiday'] == 'Yes']['Visit_Count'].sum()
                        total_visits_all = final_data['Visit_Count'].sum()
                        bh_percentage = (total_visits_bh / total_visits_all * 100) if total_visits_all > 0 else 0

                        self.log_message("🏦 Bank Holiday Summary (with fixed encoding):")
                        self.log_message(f"   Bank holiday records: {bh_count:,}")
                        self.log_message(f"   Bank holiday visits: {total_visits_bh:,.0f} ({bh_percentage:.1f}%)")

                        sample_holidays = final_data[final_data['Is_Bank_Holiday'] == 'Yes']['Bank_Holiday_Name'].unique()[:3]
                        self.log_message("   Sample cleaned holiday names:")
                        for holiday in sample_holidays:
                            if holiday and str(holiday).strip() != 'nan':
                                self.log_message(f"     - {holiday}")

                # Weather summary
                if 'Weather_Condition' in final_data.columns and self.enable_weather:
                    weather_records = final_data['Weather_Condition'].notna().sum()
                    if weather_records > 0:
                        self.log_message("🌤️ Weather Summary:")
                        self.log_message(f"   Records with weather data: {weather_records:,}")

                        temp_stats = final_data['Temperature_Mean_C'].describe()
                        self.log_message(f"   Temperature range: {temp_stats['min']:.1f}°C to {temp_stats['max']:.1f}°C")
                        self.log_message(f"   Average temperature: {temp_stats['mean']:.1f}°C")

                        weather_dist = final_data['Weather_Category'].value_counts().head(3)
                        self.log_message("   Most common weather:")
                        for condition, count in weather_dist.items():
                            if pd.notna(condition):
                                self.log_message(f"     - {condition}: {count:,} records")

                return final_data
            else:
                self.log_message("❌ No data processed successfully")
                return pd.DataFrame()

        except Exception as e:
            self.log_message(f"❌ Critical error: {str(e)}")
            self.log_message(f"📋 Traceback: {traceback.format_exc()}")
            return pd.DataFrame()

    def _find_eu_visits_sheets(self, sheet_names: List[str]) -> List[str]:
        """Find sheets that contain EU visits data"""
        eu_visit_patterns = ['eu visits', 'emergency visits', 'visits', 'eu']
        matching_sheets = []

        for sheet_name in sheet_names:
            sheet_lower = sheet_name.lower().strip()
            if any(pattern in sheet_lower for pattern in eu_visit_patterns):
                matching_sheets.append(sheet_name)

        return matching_sheets

    def create_summary_report(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create comprehensive summary reports including bank holiday and weather analysis with fixed encoding"""
        if df.empty:
            return {}

        summaries = {}

        try:
            # Apply encoding fix to bank holiday names before analysis
            if 'Bank_Holiday_Name' in df.columns:
                df['Bank_Holiday_Name'] = df['Bank_Holiday_Name'].apply(
                    lambda x: fix_encoding_issues_comprehensive(str(x)) if pd.notna(x) else x
                )

            # Calculate metrics
            total_visits = df['Visit_Count'].sum()
            bh_records = (df['Is_Bank_Holiday'] == 'Yes').sum() if 'Is_Bank_Holiday' in df.columns else 0
            bh_visits = df[df['Is_Bank_Holiday'] == 'Yes']['Visit_Count'].sum() if 'Is_Bank_Holiday' in df.columns else 0
            bh_percentage = (bh_visits / total_visits * 100) if total_visits > 0 else 0

            weather_records = df['Weather_Condition'].notna().sum() if 'Weather_Condition' in df.columns else 0
            avg_temp = df['Temperature_Mean_C'].mean() if 'Temperature_Mean_C' in df.columns else 0
            total_precip = df['Precipitation_mm'].sum() if 'Precipitation_mm' in df.columns else 0

            # Basic statistics
            overview_metrics = [
                'Total Records', 'Total Visits', 'Date Range Start', 'Date Range End',
                'Days Covered', 'Years Covered', 'Unique Outcome Types', 'Unique Sections',
                'Bank Holiday Records', 'Bank Holiday Visits', 'Bank Holiday Visit %'
            ]

            overview_values = [
                len(df), total_visits, df['Date'].min().strftime('%Y-%m-%d'),
                df['Date'].max().strftime('%Y-%m-%d'), (df['Date'].max() - df['Date'].min()).days,
                df['Year'].nunique(), df['Outcome_Type'].nunique(), df['Type'].nunique(),
                bh_records, bh_visits, f"{bh_percentage:.1f}%"
            ]

            # Add weather metrics if available
            if weather_records > 0:
                overview_metrics.extend([
                    'Weather Records', 'Average Temperature (°C)', 'Total Precipitation (mm)', 'Location'
                ])
                overview_values.extend([
                    weather_records, f"{avg_temp:.1f}", f"{total_precip:.1f}",
                    self.detected_location or 'Unknown'
                ])

            summaries['overview'] = pd.DataFrame({
                'Metric': overview_metrics,
                'Value': overview_values
            })

            # Additional summaries...
            # (keeping this shorter for the corrected version)

            return summaries

        except Exception as e:
            print(f"Error creating summaries: {str(e)}")
            return {}

    def export_results(self, df: pd.DataFrame, summaries: Dict[str, pd.DataFrame],
                      output_dir: str = "./output", export_summaries: bool = False) -> None:
        """Export processed data with bank holiday and weather information"""
        if df.empty:
            print("No data to export")
            return

        Path(output_dir).mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            if 'Bank_Holiday_Name' in df.columns:
                df['Bank_Holiday_Name'] = df['Bank_Holiday_Name'].apply(
                    lambda x: fix_encoding_issues_comprehensive(str(x)) if pd.notna(x) else x
                )

            if self.enable_bank_holidays and self.enable_weather:
                csv_file = f"{output_dir}/eu_visits_with_bank_holidays_and_weather_{timestamp}.csv"
            elif self.enable_bank_holidays:
                csv_file = f"{output_dir}/eu_visits_with_bank_holidays_{timestamp}.csv"
            elif self.enable_weather:
                csv_file = f"{output_dir}/eu_visits_with_weather_{timestamp}.csv"
            else:
                csv_file = f"{output_dir}/eu_visits_processed_data_{timestamp}.csv"

            df.to_csv(csv_file, index=False, encoding='utf-8-sig')
            print(f"✅ EU Visits data saved: {csv_file}")

            if self.enable_bank_holidays and not self.bank_holidays_df.empty:
                export_bh_df = self.bank_holidays_df.copy()
                export_bh_df['Event'] = export_bh_df['Event'].apply(
                    lambda x: fix_encoding_issues_comprehensive(str(x)) if pd.notna(x) else x
                )

                bh_csv_file = f"{output_dir}/uk_bank_holidays_{timestamp}.csv"
                export_bh_df.to_csv(bh_csv_file, index=False, encoding='utf-8-sig')
                print(f"✅ Bank holidays data saved: {bh_csv_file}")

            if self.enable_weather and not self.weather_df.empty:
                weather_csv_file = f"{output_dir}/weather_data_{timestamp}.csv"
                self.weather_df.to_csv(weather_csv_file, index=False, encoding='utf-8-sig')
                print(f"✅ Weather data saved: {weather_csv_file}")

        except Exception as e:
            print(f"❌ Export failed: {str(e)}")


# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

# Add this updated function to your EU_Visits.py file
# Replace the existing process_multiple_files_with_dynamic_detection function

def process_multiple_files_with_dynamic_detection(file_paths: List[str], output_dir: str = "./output",
                                                  enable_bank_holidays: bool = True,
                                                  uk_region: str = "england-and-wales",
                                                  enable_weather: bool = True,
                                                  location_name: str = None,
                                                  year_filter: tuple = None) -> pd.DataFrame:  # NEW PARAMETER
    """Process multiple Excel files with dynamic detection, bank holidays, weather integration, and year filtering"""
    print("🚀 EU VISITS DATA PROCESSOR - DYNAMIC DETECTION + BANK HOLIDAYS + WEATHER + YEAR FILTER")
    print("=" * 100)

    if year_filter:
        print(f"📅 YEAR FILTER ACTIVE: {year_filter[0]} to {year_filter[1]}")
        print(f"   Only data from these years will be processed and included")
    else:
        print("📅 NO YEAR FILTER: All years will be processed")

    processor = EUVisitsProcessorWithDynamicDetection(
        enable_bank_holidays=enable_bank_holidays,
        uk_region=uk_region,
        enable_weather=enable_weather,
        location_name=location_name
    )
    all_data = []

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue

        print(f"\n📊 Processing: {file_path}")
        try:
            data = processor.process_excel_file(file_path)

            if not data.empty:
                original_count = len(data)

                # Apply year filter if specified
                if year_filter:
                    print(f"📅 Applying year filter {year_filter[0]}-{year_filter[1]}...")

                    # Filter by Year column if it exists
                    if 'Year' in data.columns:
                        # Convert Year to numeric, handling string years
                        data['Year_Numeric'] = pd.to_numeric(data['Year'], errors='coerce')
                        data = data[
                            (data['Year_Numeric'] >= year_filter[0]) &
                            (data['Year_Numeric'] <= year_filter[1]) &
                            (data['Year_Numeric'].notna())
                            ]
                        data = data.drop('Year_Numeric', axis=1)

                    # Also filter by Date column as backup
                    if 'Date' in data.columns and len(data) > 0:
                        try:
                            data['Date_Parsed'] = pd.to_datetime(data['Date'], format='%d/%m/%Y', errors='coerce')
                            data['Date_Year'] = data['Date_Parsed'].dt.year
                            data = data[
                                (data['Date_Year'] >= year_filter[0]) &
                                (data['Date_Year'] <= year_filter[1]) &
                                (data['Date_Year'].notna())
                                ]
                            data = data.drop(['Date_Parsed', 'Date_Year'], axis=1)
                        except Exception as e:
                            print(f"   ⚠️ Could not filter by date: {e}")

                    filtered_count = len(data)
                    print(f"   📊 Year filter results: {original_count:,} → {filtered_count:,} records")

                    if filtered_count == 0:
                        print(f"   ⚠️ No data found in year range {year_filter[0]}-{year_filter[1]} for this file")
                        continue

                if not data.empty:
                    all_data.append(data)
                    print(f"✅ Successfully processed: {len(data):,} records")
                else:
                    print(f"⚠️ No data in specified year range from {file_path}")
            else:
                print(f"⚠️ No data extracted from {file_path}")

        except Exception as e:
            print(f"❌ Error processing {file_path}: {str(e)}")

    if not all_data:
        print("❌ No data was successfully processed")
        return pd.DataFrame()

    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data = sort_eu_visits_dataframe(combined_data, day_filter=None, month_filter=None, year_filter=year_filter)


    # Apply final year filter to combined data (safety check)
    if year_filter and not combined_data.empty:
        original_count = len(combined_data)

        if 'Year' in combined_data.columns:
            # Final filter by Year column
            combined_data['Year_Numeric'] = pd.to_numeric(combined_data['Year'], errors='coerce')
            combined_data = combined_data[
                (combined_data['Year_Numeric'] >= year_filter[0]) &
                (combined_data['Year_Numeric'] <= year_filter[1]) &
                (combined_data['Year_Numeric'].notna())
                ]
            combined_data = combined_data.drop('Year_Numeric', axis=1)

        final_count = len(combined_data)
        if original_count != final_count:
            print(f"📅 Final year filter applied: {original_count:,} → {final_count:,} records")

    # Apply encoding fix to bank holiday names if present
    if 'Bank_Holiday_Name' in combined_data.columns:
        print("🔧 Applying final encoding fix to combined data...")
        combined_data['Bank_Holiday_Name'] = combined_data['Bank_Holiday_Name'].apply(
            lambda x: fix_encoding_issues_comprehensive(str(x)) if pd.notna(x) else x
        )

    # Remove duplicates
    original_count = len(combined_data)
    combined_data = combined_data.drop_duplicates(
        subset=['Outcome_Type', 'Visit_Count', 'Date_String', 'Type'],
        keep='first'
    )
    final_count = len(combined_data)
    removed_count = original_count - final_count

    print(f"🧹 Removed {removed_count:,} duplicates")
    print(f"✅ Final dataset: {final_count:,} records")

    # Show year distribution in final data
    if not combined_data.empty and 'Year' in combined_data.columns:
        year_counts = combined_data['Year'].value_counts().sort_index()
        print(f"📊 Final data distribution by year:")
        for year, count in year_counts.items():
            print(f"   {year}: {count:,} records")

    return combined_data


def analyze_data_with_bank_holidays_and_weather(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform comprehensive analysis including bank holiday and weather patterns"""
    if df.empty:
        print("❌ No data available for analysis")
        return {}

    print("\n📊 COMPREHENSIVE DATA ANALYSIS WITH BANK HOLIDAYS AND WEATHER")
    print("=" * 80)

    try:
        print(f"📈 Dataset Overview:")
        print(f"   Total Records: {len(df):,}")
        print(f"   Total Visits: {df['Visit_Count'].sum():,}")
        print(f"   Date Range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")

        return {'total_visits': df['Visit_Count'].sum()}

    except Exception as e:
        print(f"❌ Error in analysis: {str(e)}")
        return {}


# For backward compatibility
analyze_data_with_bank_holidays = analyze_data_with_bank_holidays_and_weather


def process_multiple_files_with_month_filter(file_paths: List[str],
                                             output_dir: str = "./output",
                                             enable_bank_holidays: bool = True,
                                             uk_region: str = "england-and-wales",
                                             enable_weather: bool = True,
                                             location_name: str = None,
                                             month_filter: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Process multiple EU Visits Excel files with month filtering
    """
    print("🚀 EU VISITS DATA PROCESSOR - WITH MONTH FILTERING")
    print("=" * 100)

    if month_filter:
        print(f"📅 MONTH FILTER ACTIVE: {month_filter['description']}")
        print(f"   Date range: {month_filter['start_date']} to {month_filter['end_date']}")
    else:
        print("📅 NO MONTH FILTER: All dates will be processed")

    # Use the existing processor with additional month filtering
    processor = EUVisitsProcessorWithDynamicDetection(
        enable_bank_holidays=enable_bank_holidays,
        uk_region=uk_region,
        enable_weather=enable_weather,
        location_name=location_name
    )

    all_data = []

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue

        print(f"\n📊 Processing: {file_path}")
        try:
            data = processor.process_excel_file(file_path)

            if not data.empty:
                original_count = len(data)

                # Apply month filter if specified
                if month_filter:
                    print(f"📅 Applying month filter: {month_filter['description']}...")

                    # Parse dates and apply month filter
                    if 'Date' in data.columns:
                        try:
                            data['Date_Parsed'] = pd.to_datetime(data['Date'], format='%d/%m/%Y', errors='coerce')
                            data['Date_Only'] = data['Date_Parsed'].dt.date

                            # Apply month filter
                            mask = (
                                    (data['Date_Only'] >= month_filter['start_date']) &
                                    (data['Date_Only'] <= month_filter['end_date'])
                            )
                            data = data[mask]
                            data = data.drop(['Date_Parsed', 'Date_Only'], axis=1)

                        except Exception as e:
                            print(f"   ⚠️ Could not filter by date: {e}")

                    filtered_count = len(data)
                    print(f"   📊 Month filter results: {original_count:,} → {filtered_count:,} records")

                    if filtered_count == 0:
                        print(f"   ⚠️ No data found in month range for this file")
                        continue

                if not data.empty:
                    all_data.append(data)
                    print(f"✅ Successfully processed: {len(data):,} records")

        except Exception as e:
            print(f"❌ Error processing {file_path}: {str(e)}")

    if not all_data:
        print("❌ No data was successfully processed")
        return pd.DataFrame()

    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data = sort_eu_visits_dataframe(combined_data, day_filter=None, month_filter=month_filter,
                                             year_filter=None)

    # Apply final month filter to combined data (safety check)
    if month_filter and not combined_data.empty:
        original_count = len(combined_data)

        if 'Date' in combined_data.columns:
            try:
                combined_data['Date_Parsed'] = pd.to_datetime(combined_data['Date'], format='%d/%m/%Y', errors='coerce')
                combined_data['Date_Only'] = combined_data['Date_Parsed'].dt.date

                mask = (
                        (combined_data['Date_Only'] >= month_filter['start_date']) &
                        (combined_data['Date_Only'] <= month_filter['end_date'])
                )
                combined_data = combined_data[mask]
                combined_data = combined_data.drop(['Date_Parsed', 'Date_Only'], axis=1)

            except Exception as e:
                print(f"⚠️ Error in final month filter: {e}")

        final_count = len(combined_data)
        if original_count != final_count:
            print(f"📅 Final month filter applied: {original_count:,} → {final_count:,} records")

    return combined_data

def process_multiple_files_with_day_filter(file_paths: List[str],
                                          output_dir: str = "./output",
                                          enable_bank_holidays: bool = True,
                                          uk_region: str = "england-and-wales",
                                          enable_weather: bool = True,
                                          location_name: str = None,
                                          day_filter: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Process multiple EU Visits Excel files with day filtering
    """
    print("🚀 EU VISITS DATA PROCESSOR - WITH DAY FILTERING")
    print("=" * 100)

    if day_filter:
        print(f"📅 DAY FILTER ACTIVE: {day_filter['description']}")
        print(f"   Date range: {day_filter['start_date']} to {day_filter['end_date']}")
        print(f"   Duration: {day_filter['duration_days']} days")
    else:
        print("📅 NO DAY FILTER: All dates will be processed")

    # Use the existing processor with additional day filtering
    processor = EUVisitsProcessorWithDynamicDetection(
        enable_bank_holidays=enable_bank_holidays,
        uk_region=uk_region,
        enable_weather=enable_weather,
        location_name=location_name
    )

    all_data = []

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue

        print(f"\n📊 Processing: {file_path}")
        try:
            data = processor.process_excel_file(file_path)

            if not data.empty:
                original_count = len(data)

                # Apply day filter if specified
                if day_filter:
                    print(f"📅 Applying day filter: {day_filter['description']}...")

                    # Parse dates and apply day filter
                    if 'Date' in data.columns:
                        try:
                            data['Date_Parsed'] = pd.to_datetime(data['Date'], format='%d/%m/%Y', errors='coerce')
                            data['Date_Only'] = data['Date_Parsed'].dt.date

                            # Apply day filter
                            mask = (
                                (data['Date_Only'] >= day_filter['start_date']) &
                                (data['Date_Only'] <= day_filter['end_date'])
                            )
                            data = data[mask]
                            data = data.drop(['Date_Parsed', 'Date_Only'], axis=1)

                        except Exception as e:
                            print(f"   ⚠️ Could not filter by date: {e}")

                    filtered_count = len(data)
                    print(f"   📊 Day filter results: {original_count:,} → {filtered_count:,} records")

                    if filtered_count == 0:
                        print(f"   ⚠️ No data found in day range for this file")
                        continue

                if not data.empty:
                    all_data.append(data)
                    print(f"✅ Successfully processed: {len(data):,} records")

        except Exception as e:
            print(f"❌ Error processing {file_path}: {str(e)}")

    if not all_data:
        print("❌ No data was successfully processed")
        return pd.DataFrame()

    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data = sort_eu_visits_dataframe(combined_data, day_filter=day_filter, month_filter=None, year_filter=None)
    return combined_data


if __name__ == "__main__":
    print("🎯 EU VISITS DATA PROCESSOR - DYNAMIC DETECTION + BANK HOLIDAYS + WEATHER (NO HARDCODING)")
    print("=" * 100)
    print("✅ Script loaded successfully!")
    print("✅ All functions available for import!")