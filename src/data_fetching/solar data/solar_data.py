import pandas as pd
import pvlib

def get_solar_data(location, start_date, end_date):
    """
    Retrieves solar energy data for a specified location and date range.
    """
    # Time zone
    times = pd.date_range(start=start_date, end=end_date, freq='H', tz='Europe/Istanbul')

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
        solar_data.to_csv(f"data/raw/solar_data_{city.lower()}.csv", index=False)
        print(f"Retrieved {len(solar_data)} records for solar data in {city}.")

if __name__ == "__main__":
    main()
