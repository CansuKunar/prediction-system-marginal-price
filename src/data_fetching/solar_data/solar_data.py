import pandas as pd
import pvlib
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env')

def get_solar_data(location, start_date, end_date):
    """
    Retrieves solar energy data for a specified location and date range.

    datetime: Indicates the date and time the data was collected. It is usually in UTC (Coordinated Universal Time) format and written as YYYY-MM-DD HH:MM:SS. This column determines the time zone of solar energy data.

    dni (Direct Normal Irradiance): Indicates the direct solar radiation falling on a vertical surface. It is usually measured in watts/square meter (W/m²). DNI indicates the amount of sunlight passing through the atmosphere and reaching the surface directly and is a critical parameter for solar energy systems.

    ghi (Global Horizontal Irradiance): Indicates the total solar radiation falling on a horizontal surface. Again measured in watts/square meter (W/m²). GHI includes the sum of direct sunlight (DNI) and diffuse sunlight (light scattered from the atmosphere). It is important for evaluating the performance of solar panels.

    solar_zenith: Indicates the angle of the sun in the sky. It is measured in degrees (°) and indicates the height of the sun relative to the surface of the earth. A value of 0° means the sun is at its zenith, while a value of 90° means the sun is parallel to the horizon. The solar zenith angle has a significant impact on the efficiency of solar energy systems.
    """

    # Time zone
    times = pd.date_range(start=start_date, end=end_date, freq='h', tz='Europe/Istanbul')

    # Solar module location
    latitude, longitude = location

    # Calculate solar radiation and temperature data
    solar_position = pvlib.solarposition.get_solarposition(times, latitude, longitude)
    dni = pvlib.irradiance.get_extra_radiation(times)  # Daily normal radiation
    ghi = dni * 0.77  # Calculate global horizontal irradiance with hypothetical rate


    solar_data = pd.DataFrame({
        'datetime': times,
        'dni': dni,
        'ghi': ghi,
        'solar_zenith': solar_position['apparent_elevation'],  # Sun's zenith angle
    })

    solar_data['datetime'] = solar_data['datetime'].dt.strftime('%Y-%m-%d %H:%M')

    return solar_data

def merge_solar_data(start_date, end_date):
    """
    Merges solar data from multiple locations into a single DataFrame,
    using datetime as the index and renaming columns appropriately.
    """
    # Location information (lat, lon)
    locations = {
        'Istanbul': (41.0082, 28.9784),
        'Izmir': (38.4192, 27.1287),
        'Ankara': (39.9334, 32.8597),
        'Antalya': (36.8842, 30.7056),
        'Bursa': (40.1826, 29.0664)
    }

    merged_data = None

    for city, location in locations.items():
        print(f"Fetching solar data for {city}...")
        solar_data = get_solar_data(location, start_date, end_date)

        # Set 'datetime' as index
        solar_data.set_index('datetime', inplace=True)

        # Rename columns
        solar_data.rename(columns={
            'dni': f'{city.lower()}_dni',
            'ghi': f'{city.lower()}_ghi',
            'solar_zenith': f'{city.lower()}_solar_zenith'
        }, inplace=True)

        # Merge with the existing DataFrame
        if merged_data is None:
            merged_data = solar_data
        else:
            merged_data = merged_data.join(solar_data, how='outer')

    # Reset index to save datetime as a column
    merged_data.reset_index(inplace=True)

    # Save to CSV
    path = os.getenv('project_path')
    merged_data.to_csv(path + "/data/raw/merged_solar_data.csv", index=False)
    print("Merged solar data saved to 'merged_solar_data.csv'.")

    return merged_data

def main():
    # Date range
    start_date = "2023-10-01"
    end_date = "2024-10-01"

    # Merge solar data
    merged_solar_data = merge_solar_data(start_date, end_date)

if __name__ == "__main__":
    main()
