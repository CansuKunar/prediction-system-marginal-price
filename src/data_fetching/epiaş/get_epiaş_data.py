from eptr2 import EPTR2
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv(dotenv_path='.env')

cred_d = {
    "username":os.getenv('EPIAS_USERNAME'),
    "password":  os.getenv('EPIAS_PASSWORD'),
    "is_test": False,
}

eptr = EPTR2(
    username=cred_d["username"], password=cred_d["password"], is_test=cred_d["is_test"]
)

# Start and end dates
start_date = "2023-10-01"
end_date = "2024-10-01"

# Data types
data_columns = {
    "mcp": ["date", "hour", "price", "priceUsd", "priceEur"], #Day Ahead Market (GÖP)
    "wap": ["date", "hour", "wap"], # Intraday Market (GİP)
    "smp": ["date", "hour", "systemMarginalPrice"], # System Marginal Price
    "smp-dir": ["date", "hour", "systemDirection", "smpDirectionId"], # System Directory
    "bpm-up": ["date", "hour", "upRegulationZeroCoded", "upRegulationDelivered", "net"], # Up Regulation Instructions (YAL)
    "bpm-down": ["date", "hour", "downRegulationZeroCoded", "downRegulationDelivered", "net"], # Down Regulation Instructions (YAT)
}


dataframes = {}

def fetch_epias_data(data_type, start_date, end_date):
    """
    Retrieves data from Epiaş for the given data_type.
    """
    return eptr.call(data_type, start_date=start_date, end_date=end_date)

for key, columns in data_columns.items():
    df = fetch_epias_data(key, start_date, end_date)
    df = pd.DataFrame(df)  # Convert incoming data to DataFrame
    df = df[columns]  # Select relevant columns
    dataframes[key] = df  # Save DataFrame to dictionary

# Consolidate data based on columns 'date' and 'hour'
merged_df = dataframes["mcp"]
for key, df in dataframes.items():
    if key != "mcp":
        merged_df = pd.merge(merged_df, df, on=["date", "hour"], how="outer")


# print(merged_df.head())

path = os.getenv('project_path')
merged_df.to_csv(path+"/data/raw/merged_epias_data.csv", index=False)