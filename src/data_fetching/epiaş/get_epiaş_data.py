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


def fetch_epias_data(data_type, start_date, end_date):
    """
    Retrieves data from Epiaş for the given data_type.
    """
    return eptr.call(data_type, start_date=start_date, end_date=end_date)

def get_merged_epias_data(start_date, end_date):
    """
    Fetches and merges EPIAŞ data for the specified date range.
    Returns a DataFrame with the merged data.
    """

    dataframes = {}

    for key, columns in data_columns.items():
        df = fetch_epias_data(key, start_date, end_date)
        df = pd.DataFrame(df)  # Convert incoming data to DataFrame

        # Convert 'date' column to datetime format, ignoring 'hour' column
        df['datetime'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d %H:%M')

        # Drop original 'date' and 'hour' columns, keep only relevant columns
        df = df[['datetime'] + columns[2:]]
        dataframes[key] = df  # Save DataFrame to dictionary

        # Merge data on 'datetime' column
    merged_df = dataframes["mcp"]
    for key, df in dataframes.items():
        if key != "mcp":
            merged_df = pd.merge(merged_df, df, on="datetime", how="outer")

    # Rename duplicate 'net' columns
    merged_df = merged_df.rename(columns={
        "net_x": "upRegulationNet",
        "net_y": "downRegulationNet"
    })

    return merged_df

def main():
    # Fetch and merge EPIAŞ data
    merged_epias_data = get_merged_epias_data(start_date, end_date)

    # Save the merged data to a CSV file
    path = os.getenv('DATA_PATH')
    merged_epias_data.to_csv(path + "/data/raw/merged_epias_data.csv", index=False)
    print("Merged EPIAŞ data saved to CSV.")

if __name__ == "__main__":
    main()