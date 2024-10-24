from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path='.env')

def fetch_and_save_holiday_data(client_secret_file, start_date, end_date, output_file):
    """
    Fetches holiday data from Google Calendar API and saves it as hourly data to a CSV file.

    Parameters:
    - client_secret_file: Path to the client secret JSON file.
    - start_date: Start date for fetching data (datetime object).
    - end_date: End date for fetching data (datetime object).
    - output_file: Path to the output CSV file.
    """
    scope = ['https://www.googleapis.com/auth/calendar.readonly']

    flow = InstalledAppFlow.from_client_secrets_file(client_secret_file, scope)
    credentials = flow.run_local_server(port=0)
    service = build('calendar', 'v3', credentials=credentials)

    calendar_id = 'tr.turkish#holiday@group.v.calendar.google.com'

    events_result = service.events().list(
        calendarId=calendar_id,
        timeMin=start_date.isoformat() + 'Z',
        timeMax=end_date.isoformat() + 'Z',
        singleEvents=True,
        orderBy='startTime'
    ).execute()
    events = events_result.get('items', [])

    holiday_dates = set()
    for event in events:
        start = event['start'].get('date')
        if start:
            holiday_dates.add(start)

    hourly_data = []
    current_time = start_date
    while current_time < end_date:
        date_str = current_time.strftime('%Y-%m-%d')
        is_holiday = date_str in holiday_dates
        hourly_data.append({
            'datetime': current_time,
            'is_holiday': is_holiday
        })
        current_time += timedelta(hours=1)

    df = pd.DataFrame(hourly_data)
    df.to_csv(output_file, index=False)

    print(f"Data saved to {output_file}")

# Example usage
CLIENT_SECRET_FILE = os.getenv('CLIENT_SECRET_FILE')
start_date = datetime(2023, 10, 1)
end_date = datetime(2024, 10, 1)
output_file = os.getenv('DATA_PATH') + '/data/raw/hourly_holiday_data.csv'

fetch_and_save_holiday_data(CLIENT_SECRET_FILE, start_date, end_date, output_file)