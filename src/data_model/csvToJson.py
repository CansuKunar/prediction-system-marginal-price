import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO

# URL of the webpage containing the link
page_url = "https://www.gov.uk/government/publications/register-of-licensed-sponsors-workers"

# Send a GET request to the webpage
response = requests.get(page_url)
response.raise_for_status()  # Check for request errors

# Parse the webpage content
soup = BeautifulSoup(response.content, 'html.parser')

# Find the link to the CSV file
link_tag = soup.find('a', class_='govuk-link gem-c-attachment__link', href=True)

if link_tag:
  csv_url = link_tag['href']
  if not csv_url.startswith('http'):
      csv_url = 'https://www.gov.uk' + csv_url  # Ensure the URL is absolute

  # Download the CSV file
  csv_response = requests.get(csv_url)
  csv_response.raise_for_status()

  # Convert CSV content to a DataFrame
  csv_content = StringIO(csv_response.text)
  df = pd.read_csv(csv_content)

  # Convert the DataFrame to JSON format
  json_file_path = "Worker_and_Temporary_Worker.json"
  df.to_json(json_file_path, orient='records', force_ascii=False)