import yfinance as yf
import pytz
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv(dotenv_path='.env')


def fill_missing_hours(data):
    """
    Fills missing hours and dates in the data:
    - Fills hours from 19:00 to 10:00 with the last available data at 18:00
    - Fills weekends with Friday's last data
    - Fills any missing days or hours with the last available data
    """
    # Create complete date range
    start_date = data.index.min()
    end_date = data.index.max()
    complete_range = pd.date_range(start=start_date.date(),
                                   end=end_date.date(),
                                   freq='D',
                                   tz='Europe/Istanbul')

    filled_data = []

    for date in complete_range:
        # Get data for current day
        day_data = data[data.index.date == date.date()].between_time('10:00', '19:00')

        if day_data.empty:
            # If no data for this day, use the last available data
            if filled_data:
                last_value = filled_data[-1].iloc[-1]
            else:
                # If this is the first day and it's empty, use the next available data
                next_available = data[data.index > date].iloc[0] if not data.empty else None
                if next_available is None:
                    continue
                last_value = next_available
        else:
            last_value = day_data.iloc[-1]

        # Create full day schedule
        business_hours = pd.date_range(start=f"{date.date()} 10:00",
                                       end=f"{date.date()} 19:00",
                                       freq='h',
                                       tz='Europe/Istanbul')

        # Fill business hours
        day_filled = pd.DataFrame(index=business_hours, columns=data.columns)
        for hour in business_hours:
            if hour in day_data.index:
                day_filled.loc[hour] = day_data.loc[hour].values
            else:
                day_filled.loc[hour] = last_value.values

        # Fill after-hours (19:00 - 09:00 next day)
        next_day = date + pd.Timedelta(days=1)
        after_hours = pd.date_range(start=f"{date.date()} 18:00",
                                    end=f"{next_day.date()} 09:00",
                                    freq='h',
                                    tz='Europe/Istanbul')[1:]

        after_hours_data = pd.DataFrame([last_value.values] * len(after_hours),
                                        index=after_hours,
                                        columns=data.columns)

        full_day_data = pd.concat([day_filled, after_hours_data])
        filled_data.append(full_day_data)

        # Fill weekends
        if date.weekday() == 4:  # Friday
            weekend_dates = pd.date_range(start=next_day,
                                          end=next_day + pd.Timedelta(days=2),
                                          freq='h',
                                          tz='Europe/Istanbul')
            weekend_data = pd.DataFrame([last_value.values] * len(weekend_dates),
                                        index=weekend_dates,
                                        columns=data.columns)
            filled_data.append(weekend_data)

    # Concatenate all data
    conc_data = pd.concat(filled_data)

    # Remove duplicates that might occur at day boundaries
    conc_data = conc_data[~conc_data.index.duplicated(keep='first')]

    # Sort index and format
    conc_data = conc_data.sort_index()
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
    # Check the time for each day
    daily_check = filled_data.groupby(filled_data.index.str[:10]).count()
    print("Number of records per day:")
    print(daily_check.head())

    # Weekday/end control
    dates = pd.to_datetime(filled_data.index)
    weekday_counts = filled_data.groupby(dates.strftime('%A')).count()
    print("\nRecords by day of week:")
    print(weekday_counts)

    # Time range control
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
    end_date = "2024-11-01"

    print("Fetching BIST100 data...")
    bist100_data = get_bist100_data(start_date, end_date)

    if not bist100_data.empty:
        print(f"Retrieved {len(bist100_data)} hourly records for BIST100.")

        filled_data = fill_missing_hours(bist100_data)
        print(filled_data.tail(20))

        # Save to CSV with Date as index
        output_file = os.path.join(output_dir, "bist100_hourly_data.csv")
        verify_filled_data(filled_data)
        filled_data.to_csv(output_file, index=True)
        print(f"Data saved to {output_file}")
    else:
        print("BIST100 data couldn't be retrieved.")


if __name__ == "__main__":
    main()