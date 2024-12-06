import pandas as pd
import pvlib


def get_solar_data(location, start_date, end_date):
    """
    Retrieves solar energy data for a specified location and date range.
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
        'dni': dni,
        'ghi': ghi,
        'solar_zenith': solar_position['apparent_elevation'],  # Sun's zenith angle
    }, index=times)

    # Format index to desired string format
    solar_data.index = solar_data.index.strftime('%Y-%m-%d %H:%M')
    solar_data.index.name = 'datetime'

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

    return merged_data


def main():
    # Date range
    start_date = "2023-10-01"
    end_date = "2024-11-02"

    # Merge solar data
    merged_solar_data = merge_solar_data(start_date, end_date)
    print(merged_solar_data.tail(5))



if __name__ == "__main__":
    main()