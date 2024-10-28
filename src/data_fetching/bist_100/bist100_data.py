import yfinance as yf
import pytz
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv(dotenv_path='.env')


def fill_missing_hours(data):
    """
    It fills the hours from 19:00 to 10:00 for each day with the data at 18:00.
    Additionally, it fills the weekend with the last available data from Friday.
    """
    filled_data = []
    grouped = data.groupby(data.index.date)

    for date, group in grouped:
        day_data = group.between_time('10:00', '18:00')

        if not day_data.empty:
            # Get the last data at 18:00
            last_value = day_data.iloc[-1]

            # Fill in between 19:00 - 09:00
            next_day = pd.Timestamp(date) + pd.Timedelta(days=1)
            times_to_fill = pd.date_range(start=f"{date} 18:00", end=f"{next_day} 09:00", freq='h',
                                          tz='Europe/Istanbul')[1:]
            fill_data = pd.DataFrame([last_value] * len(times_to_fill), index=times_to_fill)

            full_day_data = pd.concat([day_data, fill_data])
            filled_data.append(full_day_data)

            # Check if the current day is Friday
            if pd.Timestamp(date).weekday() == 4:  # 4 corresponds to Friday
                # Fill Saturday and Sunday with Friday's last data
                weekend_dates = pd.date_range(start=f"{next_day}", end=f"{next_day + pd.Timedelta(days=2)} 09:00", freq='h',
                                              tz='Europe/Istanbul')
                weekend_fill_data = pd.DataFrame([last_value] * len(weekend_dates), index=weekend_dates)
                filled_data.append(weekend_fill_data)

    conc_data = pd.concat(filled_data)
    # Format index to desired string format and set name to 'Date'
    conc_data.index = conc_data.index.strftime('%Y-%m-%d %H:%M')
    conc_data.index.name = 'datetime'
    return conc_data


def get_bist100_data(start_date, end_date):
    """
    It pulls BIST100 data using Yahoo Finance API.
    It returns daily data within the specified date range.
    """
    # BIST100 symbol
    symbol = 'XU100.IS'  # Yahoo Finance symbol for BIST100
    data = yf.download(symbol, start=start_date, end=end_date, interval='1h')

    istanbul_tz = pytz.timezone('Europe/Istanbul')
    data.index = data.index.tz_convert(istanbul_tz)

    data.index = data.index.ceil('h')
    data = data[(data.index >= pd.Timestamp(start_date, tz=istanbul_tz)) &
                (data.index <= pd.Timestamp(end_date, tz=istanbul_tz))]

    # Remove ticker level from columns
    data.columns = data.columns.droplevel('Ticker')

    return data

def verify_filled_data(filled_data):
    # Her gün için saat kontrolü
    daily_check = filled_data.groupby(filled_data.index.str[:10]).count()
    print("Number of records per day:")
    print(daily_check.head())

    # Hafta içi/sonu kontrolü
    dates = pd.to_datetime(filled_data.index)
    weekday_counts = filled_data.groupby(dates.strftime('%A')).count()
    print("\nRecords by day of week:")
    print(weekday_counts)

    # Saat aralığı kontrolü
    hours = pd.to_datetime(filled_data.index).hour
    hour_counts = filled_data.groupby(hours).count()
    print("\nRecords by hour:")
    print(hour_counts)

def main():
    path = os.getenv('DATA_PATH')
    if not path:
        print("DATA_PATH environment variable is not set.")
        return

    # Ensure the directory exists
    output_dir = os.path.join(path, "data/raw")
    os.makedirs(output_dir, exist_ok=True)

    start_date = "2023-9-28"
    end_date = "2024-10-01"

    print("Fetching BIST100 data...")
    bist100_data = get_bist100_data(start_date, end_date)

    if not bist100_data.empty:
        print(f"Retrieved {len(bist100_data)} hourly records for BIST100.")

        filled_data = fill_missing_hours(bist100_data)

        # Save to CSV with Date as index
        output_file = os.path.join(output_dir, "bist100_hourly_data.csv")
        filled_data.to_csv(output_file, index=True)
        print(f"Data saved to {output_file}")

        # Display first few rows to verify format
        print("\nFirst few rows of the data:")
        print(filled_data.head())
    else:
        print("BIST100 data couldn't be retrieved.")


if __name__ == "__main__":
    main()