from meteostat import Point, Hourly

def get_hourly_weather_data(latitude, longitude, start_date, end_date):
    """
    Fetches hourly weather data for a year from the Meteostat API for the specified coordinates.
    Returns wind speed, temperature, and humidity information.
    """
    # Pulling weather data
    location = Point(latitude, longitude)  # Determine coordinates
    data = Hourly(location, start_date, end_date)
    data = data.fetch()

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
            print(f"Retrieved {len(weather_data)} hourly records for {city}.")
            weather_data.to_csv(f"data/raw/{city}_hourly_weather.csv", index=True)
        else:
            print(f"Weather data couldn't be retrieved for {city}.")

if __name__ == "__main__":
    main()
