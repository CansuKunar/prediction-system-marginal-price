import yfinance as yf
import pytz
import pandas as pd
from dotenv import load_dotenv
import os



load_dotenv(dotenv_path='.env')

def fill_missing_hours(data):
    """
    Her gün için 19:00'dan 10:00'a kadar olan saatleri
    18:00'daki veri ile doldurur.
    """
    filled_data = []
    # Günlük gruplama
    grouped = data.groupby(data.index.date)

    for date, group in grouped:
        # 10:00 - 18:00 arasındaki verileri al
        day_data = group.between_time('10:00', '18:00')

        if not day_data.empty:
            # 18:00'daki son veriyi al
            last_value = day_data.iloc[-1]

            # 19:00 - 09:00 arasını doldur
            next_day = pd.Timestamp(date) + pd.Timedelta(days=1)
            times_to_fill = pd.date_range(start=f"{date} 18:00", end=f"{next_day} 09:00", freq='h',
                                          tz='Europe/Istanbul')[1:]
            fill_data = pd.DataFrame([last_value] * len(times_to_fill), index=times_to_fill)

            # Günün verilerini birleştir
            full_day_data = pd.concat([day_data, fill_data])
            filled_data.append(full_day_data)

    # Tüm günleri birleştir
    return pd.concat(filled_data)

def get_bist30_data(start_date, end_date):
  """
  BIST100 verilerini Yahoo Finance API'si kullanarak çeker.
  Belirtilen tarih aralığında saatlik veri döner.
  """
  # BIST300 sembolü
  symbol = 'XU030.IS'  # BIST100 için Yahoo Finance sembolü

  # Veri çekme
  data = yf.download(symbol, start=start_date, end=end_date, interval='1h')

  # Zaman dilimini Istanbul'a çevir
  istanbul_tz = pytz.timezone('Europe/Istanbul')
  data.index = data.index.tz_convert(istanbul_tz)

  # Zaman damgalarını yukarı yuvarla
  data.index = data.index.ceil('h')

  return data


def main():

  path = os.getenv('DATA_PATH')

  start_date = "2023-10-01"
  end_date = "2024-10-01"

  print("Fetching BIST30 data...")
  bist30_data = get_bist30_data(start_date, end_date)
  if not bist30_data.empty:
      print(f"Retrieved {len(bist30_data)} hourly records for BIST30.")
      # Verileri data/raw klasörüne kaydet
      filled_data = fill_missing_hours(bist30_data)
      filled_data.to_csv(path+"/data/raw/bist30_hourly_data.csv", index=True)
      print("Data saved to bist30_hourly_data.csv")
  else:
      print("BIST30 data couldn't be retrieved.")

main()