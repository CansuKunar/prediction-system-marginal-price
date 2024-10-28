import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path='.env')

# Import functions from the uploaded documents
from data_fetching.bist_30 import bist_30_data as bist_30
from data_fetching.bist_100 import bist100_data as bist_100
from data_fetching.weather import weather_data as weather
from data_fetching.epiaş import get_epiaş_data as epia
from data_fetching.solar_data import solar_data as solar
from data_fetching.public_holidays import get_holidays as ghol

def main():
  # Define date range
  extended_start_date = "2023-09-01"
  start_date_ep = "2023-10-01"
  end_date_ep = "2024-10-01"
  extended_end_date = "2024-10-10"

  # Fetch BIST30 data
  print("Fetching BIST30 data...")
  bist30_data = bist_30.get_bist30_data(extended_start_date, extended_end_date)
  bist30_data = bist_30.fill_missing_hours(bist30_data)
  print(f"BIST30 index type: {type(bist30_data.index)}")

  # Fetch BIST100 data
  print("Fetching BIST100 data...")
  bist100_data = bist_100.get_bist100_data(extended_start_date, extended_end_date)
  bist100_data = bist_100.fill_missing_hours(bist100_data)
  print(f"BIST100 index type: {type(bist100_data.index)}")

  # Fetch EPIAS data
  print("Fetching EPIAS data...")
  merged_epias_data = epia.get_merged_epias_data(start_date_ep, end_date_ep)
  print(f'epiaş: {len(merged_epias_data)}')
  print(f"EPIAS index type: {type(merged_epias_data.index)}")

  # Fetch holiday data
  print("Fetching holiday data...")
  CLIENT_SECRET_FILE = os.getenv('CLIENT_SECRET_FILE')
  holiday_data = ghol.fetch_holiday_data(CLIENT_SECRET_FILE, datetime(2023, 9,1), datetime(2024, 10, 10))

  # Fetch solar data
  merged_solar_data = solar.merge_solar_data(extended_start_date, extended_end_date)
  print(f"Solar index type: {type(merged_solar_data.index)}")

  # Fetch solar data

  merged_weather_data = weather.merge_weather_data(extended_start_date, extended_end_date)
  print(f"Weather index type: {type(merged_weather_data.index)}")
  # Merge all data
  print("Merging all datasets...")

  # Create dictionary of dataframes and check for None values
  dfs = {
      'bist30': bist30_data,
      'bist100': bist100_data,
      'epias': merged_epias_data,
      'holiday': holiday_data,
      'solar': merged_solar_data,
      'weather': merged_weather_data
  }

  # Convert indices to datetime and print information
  print("\nConverting indices to datetime format...")
  for name, df in dfs.items():
      if df is not None:
          try:
              # EPIAS için özel işlem
              if name == 'epias':
                  if 'datetime' in df.columns:
                      df.set_index('datetime', inplace=True)
                  elif 'date' in df.columns:
                      df.set_index('date', inplace=True)

              # Index'i datetime'a çevir
              df.index = pd.to_datetime(df.index)
              df.sort_index(inplace=True)
              dfs[name] = df

              print(f"\n{name} DataFrame:")
              print(f"Index type after conversion: {type(df.index)}")
              print(f"Date range: {df.index.min()} to {df.index.max()}")
              print(f"Shape: {df.shape}")
          except Exception as e:
              print(f"Error converting {name} index: {e}")

  # Check for None values and create valid_dfs
  print("\nChecking data availability:")
  valid_dfs = {}
  for name, df in dfs.items():
      if df is None:
          print(f"Warning: {name} data is None")
      else:
          print(f"Success: {name} data is available with shape {df.shape}")
          valid_dfs[name] = df

  if not valid_dfs:
      raise ValueError("No valid DataFrames available for merging!")

  # Start with the first available DataFrame
  first_df_name = list(valid_dfs.keys())[0]
  all_data = valid_dfs[first_df_name].copy()
  print(f"\nStarting merge with {first_df_name} data")

  # Merge with other dataframes
  for df_name, df in valid_dfs.items():
      if df_name != first_df_name:
          print(f"Merging with {df_name} data...")
          try:
              all_data = pd.merge(
                  all_data,
                  df,
                  left_index=True,
                  right_index=True,
                  how='outer'
              )
              print(f"Successfully merged {df_name}")
          except Exception as e:
              print(f"Error merging {df_name}: {str(e)}")

  # Handle any duplicate columns

  # Sort the final dataframe by datetime index
  all_data.sort_index(inplace=True)

  # Print final dataset information
  print("\nFinal Dataset Information:")
  print("-" * 80)
  print(f"Total rows: {len(all_data)}")
  print(f"Start date: {all_data.index.min()}")
  print(f"End date: {all_data.index.max()}")
  print(f"Total columns: {len(all_data.columns)}")
  print("-" * 80)
  print("\nFiltering data by date range...")
  print(f"Before filtering: {len(all_data)} rows")

  try:
      # Filter by date range
      all_data = all_data[
          (all_data.index >= start_date_ep) &
          (all_data.index <= end_date_ep)
          ]

      # Check index and show duplicate values
      duplicated_times = all_data.index.duplicated(keep=False)
      if duplicated_times.any():
          print("\nWarning: Found duplicate timestamps:")
          print(all_data[duplicated_times].index.value_counts().head())

          # Remove duplicate values (keep first value)
          all_data = all_data[~all_data.index.duplicated(keep='first')]

      # Check hourly frequency
      expected_index = pd.date_range(start=start_date_ep, end=end_date_ep, freq='H')
      missing_times = expected_index.difference(all_data.index)
      extra_times = all_data.index.difference(expected_index)

      if len(missing_times) > 0:
          print("\nMissing timestamps:")
          print(f"Total missing: {len(missing_times)}")
          print("First few missing times:", missing_times[:5])

      if len(extra_times) > 0:
          print("\nExtra timestamps:")
          print(f"Total extra: {len(extra_times)}")
          print("First few extra times:", extra_times[:5])

      # Set correct hourly index
      all_data = all_data.reindex(expected_index)

      print(f"\nAfter fixing timestamps:")
      print(f"Total rows: {len(all_data)}")
      print(f"Date range: from {all_data.index.min()} to {all_data.index.max()}")

  except Exception as e:
      print(f"Error during date filtering: {str(e)}")

  # Print final dataset information
  print("\nFinal Dataset Information:")
  print("-" * 80)
  print(f"Total rows: {len(all_data)}")
  print(f"Start date: {all_data.index.min()}")
  print(f"End date: {all_data.index.max()}")
  print(f"Total columns: {len(all_data.columns)}")
  print("-" * 80)

  # Save to CSV
  path = os.getenv('project_path')
  all_data.to_csv(os.path.join(path, "data/raw/combined_data.csv"))
  print("All data saved to combined_data.csv")



if __name__ == "__main__":
    main()