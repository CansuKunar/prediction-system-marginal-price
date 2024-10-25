from meteostat import Point, Hourly
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env')

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

    data['date'] = data.index.date
    data['hour'] = data.index.time
    data.reset_index(drop=True, inplace=True)

    data = data[['date', 'hour', 'temp', 'dwpt', 'rhum', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres', 'tsun', 'coco']]

    return data

def main():
    # Coordinates of cities
    cities = {
        'Istanbul': (41.0082, 28.9784),
        'Izmir': (38.4192, 27.1287),
        'Ankara': (39.9334, 32.8597),
        'Antalya': (36.8969, 30.7133),
        'Bursa': (40.1826, 29.0668)
    }

    start_date = "2023-10-01"
    end_date = "2024-10-01"

    for city, (lat, lon) in cities.items():
        print(f"Fetching weather data for {city}...")
        weather_data = get_hourly_weather_data(lat, lon, start_date, end_date)

        if not weather_data.empty:
            path = os.getenv('project_path')
            print(f"Retrieved {len(weather_data)} hourly records for {city}.")
            weather_data.to_csv(path + f"/data/raw/{city}_hourly_weather.csv", index=False)
        else:
            print(f"Weather data couldn't be retrieved for {city}.")

if __name__ == "__main__":
    main()

