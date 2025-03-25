import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime
from tqdm import tqdm
import time
import os
import json

# Configuration
FILE_NAME = 'taxis_dataset.csv'
CHECKPOINT_FILE = 'weather_progress_checkpoint.json'
OUTPUT_FILE = 'taxis_dataset_with_features.csv'
CACHE_DIR = '.weather_cache'

# Holiday data (2015-2016)
NY_HOLIDAYS = {
    '2015-01-01': "New Year's Day",
    '2015-01-19': "Martin Luther King Jr. Day",
    '2015-02-16': "Presidents' Day",
    '2015-05-25': "Memorial Day",
    '2015-07-04': "Independence Day",
    '2015-09-07': "Labor Day",
    '2015-10-12': "Columbus Day",
    '2015-11-11': "Veterans Day",
    '2015-11-26': "Thanksgiving Day",
    '2015-12-25': "Christmas Day",
    '2016-01-01': "New Year's Day",
    '2016-01-18': "Martin Luther King Jr. Day",
    '2016-02-15': "Presidents' Day",
    '2016-05-30': "Memorial Day",
    '2016-07-04': "Independence Day",
    '2016-09-05': "Labor Day",
    '2016-10-10': "Columbus Day",
    '2016-11-11': "Veterans Day",
    '2016-11-24': "Thanksgiving Day",
    '2016-12-25': "Christmas Day"
}

# Setup API client with cache
cache_session = requests_cache.CachedSession(CACHE_DIR, expire_after=86400)  # 24h cache
retry_session = retry(cache_session, retries=5, backoff_factor=0.3)
openmeteo = openmeteo_requests.Client(session=retry_session)

def safe_int_convert(value, default=1):
    """Safely convert value to integer with fallback"""
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default

def load_data():
    """Load and prepare the dataset"""
    print("📄 Loading dataset...")
    df = pd.read_csv(FILE_NAME)
    
    # Validate columns
    required_cols = ['year', 'month', 'day', 'hour', 'pickup_longitude', 'pickup_latitude']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert and clean date columns
    df['year'] = df['year'].apply(safe_int_convert, default=2015)
    df['month'] = df['month'].apply(safe_int_convert)
    df['day'] = df['day'].apply(safe_int_convert)
    df['hour'] = df['hour'].apply(safe_int_convert, default=0)
    
    # Validate date ranges
    df['month'] = df['month'].clip(1, 12)
    df['day'] = df['day'].clip(1, 31)
    
    # Create date string with robust formatting
    def make_date_str(row):
        try:
            return f"{int(row['year'])}-{int(row['month']):02d}-{int(row['day']):02d}"
        except:
            return "2015-01-01"  # Fallback date
    
    df['date_str'] = df.apply(make_date_str, axis=1)
    
    # Round coordinates
    df['pickup_lat_rounded'] = df['pickup_latitude'].round(1)
    df['pickup_lon_rounded'] = df['pickup_longitude'].round(1)
    
    # Mark holidays
    df['is_holiday'] = df['date_str'].apply(lambda d: 1 if d in NY_HOLIDAYS else 0)
    
    # Initialize weather columns
    df['temperature'] = None
    df['precipitation'] = None
    
    return df

def load_checkpoint():
    """Load processed combinations from checkpoint file"""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                return set(json.load(f))
        except:
            return set()
    return set()

def save_checkpoint(processed):
    """Save progress to checkpoint file"""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(list(processed), f)

def fetch_weather(date_str, lat, lon):
    """Fetch weather data from API"""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": "temperature_2m,precipitation",
        "timezone": "America/New_York"
    }
    
    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        hourly = response.Hourly()
        
        weather_data = pd.DataFrame({
            "hour": range(24),
            "temperature": hourly.Variables(0).ValuesAsNumpy(),
            "precipitation": hourly.Variables(1).ValuesAsNumpy()
        })
        
        return weather_data
    
    except Exception as e:
        print(f"❌ Error fetching weather for {date_str} ({lat:.1f},{lon:.1f}): {str(e)}")
        return None

def process_weather_data(df):
    """Process weather data with checkpoint functionality"""
    unique_combos = df[['date_str', 'pickup_lat_rounded', 'pickup_lon_rounded']].drop_duplicates()
    processed = load_checkpoint()
    
    # Filter already processed combinations
    pending_combos = []
    for _, row in unique_combos.iterrows():
        combo_key = (row['date_str'], float(row['pickup_lat_rounded']), float(row['pickup_lon_rounded']))
        if combo_key not in processed:
            pending_combos.append(row)
    
    if not pending_combos:
        print("✅ All weather data already processed!")
        return df
    
    print(f"⛅ Processing {len(pending_combos)} weather data combinations...")
    
    progress_bar = tqdm(pending_combos)
    for row in progress_bar:
        date_str = row['date_str']
        lat = float(row['pickup_lat_rounded'])
        lon = float(row['pickup_lon_rounded'])
        combo_key = (date_str, lat, lon)
        
        # Update progress bar description
        progress_bar.set_description(f"Processing {date_str} ({lat:.1f}, {lon:.1f})")
        
        # Fetch weather data
        weather_df = fetch_weather(date_str, lat, lon)

        if weather_df is None:
            continue
        
        # Apply weather data to matching rows
        mask = (
            (df['date_str'] == date_str) &
            (df['pickup_lat_rounded'] == row['pickup_lat_rounded']) &
            (df['pickup_lon_rounded'] == row['pickup_lon_rounded'])
        )
        
        for idx in df[mask].index:
            hour = df.at[idx, 'hour']
            weather = weather_df[weather_df['hour'] == hour]
            if not weather.empty:
                df.at[idx, 'temperature'] = weather['temperature'].values[0]
                df.at[idx, 'precipitation'] = weather['precipitation'].values[0]
        
        # Update checkpoint
        processed.add(combo_key)
        save_checkpoint(processed)
        
        # Rate limiting
        time.sleep(0.5)
    
    return df

def main():
    try:
        # Load and prepare data
        df = load_data()
        
        # Process weather data with checkpoint
        df = process_weather_data(df)
        
        # Clean up
        df.drop(['date_str', 'pickup_lat_rounded', 'pickup_lon_rounded'], axis=1, inplace=True)
        
        # Save final output
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✅ Final data saved to: {OUTPUT_FILE}")
        
        # Remove checkpoint if completed
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            
    except Exception as e:
        print(f"\n⚠️ Script interrupted: {str(e)}")
        print("Progress has been saved. Rerun to continue where you left off.")
        raise

if __name__ == "__main__":
    main()