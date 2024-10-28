from meteostat import Point, Hourly
from datetime import datetime
import pandas as pd


def get_hourly_weather_data(latitude, longitude, start_date, end_date):
    """
    Fetches hourly weather data for a year from the Meteostat API for the specified coordinates.
    Returns wind speed, temperature, and humidity information.

    time: Indicates the time period when weather data is collected. It is usually in date and time format (e.g. YYYY-MM-DD HH:MM:SS).

    temp: Indicates the air temperature. It is usually measured in Celsius or Fahrenheit. The temperature reflects the perceived temperature in the environment.

    dwpt (Dew Point Temperature): Indicates the dew point temperature. It is the temperature at which moisture vapor begins to condense when the air reaches this temperature. The dew point indicates the amount of humidity in the air; higher values ​​indicate more humid weather conditions.

    rhum (Relative Humidity): Indicates the relative humidity rate. It is the ratio of the current moisture content of the air to the maximum moisture content that the air can hold at that temperature. It is expressed as a percentage (%); 100% means that the air is completely saturated.

    prcp (Precipitation): Indicates the amount of precipitation. It is usually measured in millimeters (mm) and indicates the amount of precipitation that falls in a certain period of time (hour, day).

    snow: Indicates snowfall. It is usually measured in millimeters or centimeters and indicates the amount of snow that has fallen in a certain period of time.

    wdir (Wind Direction): Indicates the direction from which the wind is coming. It is expressed in degrees (°); 0° represents the north, 90° east, 180° south and 270° west.

    wspd (Wind Speed): Indicates the wind speed. It is usually measured in meters per second (m/s), kilometers per hour (km/h) or miles per hour (mph).

    wpgt (Wind Gust): Indicates the maximum speed of the wind at the moment of the gusts (strong wind) and is usually measured in the same units (m/s, km/h, mph).
    pres (Pressure): Indicates the air pressure. It is usually measured in millibars (hPa) or inches of mercury (inHg). Air pressure indicates the weight of the atmosphere and is an important factor affecting the weather.

    tsun (Sunshine): Indicates the duration of sunlight. It is usually expressed in hours or minutes. It indicates the total amount of sunlight received in a certain period of time (e.g. daily).

    coco (Cloud Cover): It indicates cloud cover. It is usually expressed as a percentage (%), octave (8 slices), or a specific scale. In weather analysis, cloud cover is an important parameter affecting the weather.
    """

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # Pulling weather data
    location = Point(latitude, longitude)  # Determine coordinates
    data = Hourly(location, start_date, end_date)
    data = data.fetch()

    data['datetime'] = data.index.strftime('%Y-%m-%d %H:%M')
    data.set_index('datetime', inplace=True)

    return data


def merge_weather_data(start_date, end_date):
    """
    Merges weather data from multiple locations into a single DataFrame.
    """
    cities = {
        'Istanbul': (41.0082, 28.9784),
        'Izmir': (38.4192, 27.1287),
        'Ankara': (39.9334, 32.8597),
        'Antalya': (36.8969, 30.7133),
        'Bursa': (40.1826, 29.0668)
    }

    merged_data = None

    for city, (lat, lon) in cities.items():
        print(f"Fetching weather data for {city}...")
        weather_data = get_hourly_weather_data(lat, lon, start_date, end_date)

        if not weather_data.empty:
            # Rename columns to include city name
            weather_data.rename(columns={
                'temp': f'{city.lower()}_temp',
                'dwpt': f'{city.lower()}_dwpt',
                'rhum': f'{city.lower()}_rhum',
                'prcp': f'{city.lower()}_prcp',
                'snow': f'{city.lower()}_snow',
                'wdir': f'{city.lower()}_wdir',
                'wspd': f'{city.lower()}_wspd',
                'wpgt': f'{city.lower()}_wpgt',
                'pres': f'{city.lower()}_pres',
                'tsun': f'{city.lower()}_tsun',
                'coco': f'{city.lower()}_coco'
            }, inplace=True)

            # Merge with the existing DataFrame
            if merged_data is None:
                merged_data = weather_data
            else:
                merged_data = merged_data.join(weather_data, how='outer')
        else:
            print(f"Weather data couldn't be retrieved for {city}.")

    # Ensure the index is properly formatted before saving
    if merged_data is not None:
        # Check if index is already datetime
        if not isinstance(merged_data.index, pd.DatetimeIndex):
            # Convert index to datetime if it's not already
            merged_data.index = pd.to_datetime(merged_data.index)

        # Format index to desired string format
        merged_data.index = merged_data.index.strftime('%Y-%m-%d %H:%M')
        merged_data.index.name = 'Date'

    return merged_data


def main():
    start_date = "2023-10-01"
    end_date = "2024-10-01"

    # Merge weather data
    merged_weather_data = merge_weather_data(start_date, end_date)


if __name__ == "_main_":
    main()