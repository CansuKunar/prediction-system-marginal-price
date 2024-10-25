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

    solar_data['date'] = solar_data['datetime'].dt.date
    solar_data['hour'] = solar_data['datetime'].dt.time
    solar_data = solar_data[['date', 'hour', 'dni', 'ghi', 'solar_zenith']]

    return solar_data

def main():
    # Location information (lat, lon)
    locations = {
        'Istanbul': (41.0082, 28.9784),
        'Izmir': (38.4192, 27.1287),
        'Ankara': (39.9334, 32.8597),
        'Antalya': (36.8842, 30.7056),
        'Bursa': (40.1826, 29.0664)
    }

    # Date range
    start_date = "2023-10-01"
    end_date = "2024-10-01"

    for city, location in locations.items():
        print(f"Fetching solar data for {city}...")
        solar_data = get_solar_data(location, start_date, end_date)

        # Save data to data/raw folder
        path = os.getenv('project_path')
        solar_data.to_csv(path + f"/data/raw/solar_data_{city.lower()}.csv", index=False)
        print(f"Retrieved {len(solar_data)} records for solar data in {city}.")

if __name__ == "__main__":
    main()
