import yfinance as yf
import pytz
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv(dotenv_path='.env')
def fill_missing_hours(data):

    """
    It fills the hours from 19:00 to 10:00 for each day with the data at 18:00.
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

    return pd.concat(filled_data)

def get_bist100_data(start_date, end_date):
    """
    It pulls BIST100 data using Yahoo Finance API.
    It returns daily data within the specified date range.
    """
    # BIST100 symbol
    symbol = 'XU100.IS'  # Yahoo Finance symbol for BIST100

    # Data extraction
    data = yf.download(symbol, start=start_date, end=end_date, interval='1h')

    # Set time zone
    istanbul_tz = pytz.timezone('Europe/Istanbul')
    data.index = data.index.tz_convert(istanbul_tz)

    data.index = data.index.ceil('h')

    return data


def main():

    path = os.getenv('DATA_PATH')
    print(path)

    start_date =  "2023-10-01"
    end_date = "2024-10-01"

    print("Fetching BIST100 data...")
    bist100_data = get_bist100_data(start_date, end_date)

    if not bist100_data.empty:
        print(f"Retrieved {len(bist100_data)} hourly records for BIST100.")

        filled_data = fill_missing_hours(bist100_data)
        filled_data.to_csv(path+"/data/raw/bist100_hourly_data.csv", index=True)
    else:
        print("BIST100 data couldn't be retrieved.")


if __name__ == "__main__":
    main()
