import yfinance as yf
import pytz
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env')

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

    path = os.getenv('project_path')

    start_date =  "2023-10-01"
    end_date = "2024-10-01"

    print("Fetching BIST100 data...")
    bist100_data = get_bist100_data(start_date, end_date)

    if not bist100_data.empty:
        print(f"Retrieved {len(bist100_data)} hourly records for BIST100.")

        bist100_data.to_csv(path+"/Data/raw/bist100_hourly_data.csv", index=True)
    else:
        print("BIST100 data couldn't be retrieved.")


if __name__ == "__main__":
    main()
