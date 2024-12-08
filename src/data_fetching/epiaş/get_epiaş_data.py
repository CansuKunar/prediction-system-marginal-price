from eptr2 import EPTR2
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv(dotenv_path='.env') 
path = r'C:\Users\Salih\Documents\GitHub\prediction-system-marginal-price'
cred_d = {
    "username":'slhmtnn06@gmail.com',
    "password":  'Salih123.',
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
    Returns a DataFrame with the merged data and handles outliers.
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

    # Set datetime as index
    merged_df.set_index('datetime', inplace=True)
    merged_df.index = pd.to_datetime(merged_df.index)

    # Detect outliers in systemMarginalPrice using IQR method
    Q1 = merged_df['systemMarginalPrice'].quantile(0.25)
    Q3 = merged_df['systemMarginalPrice'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR 
    upper_bound = Q3 + 1.5 * IQR

    # Create a copy of the original data
    data_cleaned = merged_df.copy()
    
    # Function to check if a value is an outlier
    def is_outlier(value):
        if pd.isna(value):
            return True
        return value < lower_bound or value > upper_bound

    # Function to validate replacement value
    def is_valid_replacement(value):
        if pd.isna(value):
            return False
        return value >= 300  # Minimum acceptable value

    # Function to find first non-outlier value in previous hours
    def find_previous_non_outlier(idx, max_hours_back=168):  # 168 hours = 7 days
        current_hour = idx.hour
        hours_back = 1
        while hours_back <= max_hours_back:
            check_time = idx - pd.Timedelta(hours=hours_back)
            if check_time in merged_df.index:
                value = merged_df.loc[check_time, 'systemMarginalPrice']
                if not is_outlier(value) and is_valid_replacement(value):
                    return value, hours_back
            hours_back += 1
        return None, None

    # Identify outliers
    outliers_mask = (
        (merged_df['systemMarginalPrice'] < lower_bound) | 
        (merged_df['systemMarginalPrice'] > upper_bound)
    )
    outliers = merged_df[outliers_mask]
    
    # Store original outliers for reporting
    original_outliers = outliers.copy()
    replacement_details = []
    
    # Process each outlier
    for idx in outliers.index:
        # Check previous and next values
        prev_time = idx - pd.Timedelta(hours=1)
        next_time = idx + pd.Timedelta(hours=1)
        
        prev_value = merged_df.loc[prev_time, 'systemMarginalPrice'] if prev_time in merged_df.index else None
        next_value = merged_df.loc[next_time, 'systemMarginalPrice'] if next_time in merged_df.index else None
        
        # Check if both neighboring values are non-outliers and valid replacements
        if (prev_value is not None and not is_outlier(prev_value) and is_valid_replacement(prev_value) and 
            next_value is not None and not is_outlier(next_value) and is_valid_replacement(next_value)):
            # Use average of neighboring values
            replacement_value = (prev_value + next_value) / 2
            replacement_type = 'Neighbor Average'
            hours_back = 'N/A'
        else:
            # Find first non-outlier value in previous hours
            replacement_value, hours_back = find_previous_non_outlier(idx)
            
            if replacement_value is not None:
                replacement_type = f'Previous {hours_back} hours'
            else:
                # If no valid previous value found, use the mean of all valid non-outlier values
                non_outlier_values = merged_df[
                    (~outliers_mask) & 
                    (merged_df['systemMarginalPrice'] >= 300)
                ]['systemMarginalPrice']
                
                if len(non_outlier_values) > 0:
                    replacement_value = non_outlier_values.mean()
                else:
                    # If no valid values found, use 300 as minimum
                    replacement_value = 300
                
                replacement_type = 'Overall Mean'
                hours_back = 'N/A'
        
        # Final check to ensure replacement value is valid
        if not is_valid_replacement(replacement_value):
            replacement_value = 300
            replacement_type = 'Minimum Value (300)'
        
        # Replace the outlier with the calculated value
        data_cleaned.loc[idx, 'systemMarginalPrice'] = replacement_value
        
        replacement_details.append({
            'datetime': idx,
            'original_value': merged_df.loc[idx, 'systemMarginalPrice'],
            'replaced_value': replacement_value,
            'replacement_type': replacement_type,
            'hours_back': hours_back,
            'prev_value': prev_value,
            'next_value': next_value
        })
    
    # Create detailed report
    outliers_report = pd.DataFrame(replacement_details)
    # Save outliers report 
    outliers_report.to_csv(path + "/data/outliers_report.csv", index=False)   
    
    # Print statistics and report
    print(f"\nOutlier Statistics for System Marginal Price:")
    print(f"Number of outliers detected: {len(original_outliers)}")
    print(f"Percentage of outliers: {(len(original_outliers)/len(merged_df))*100:.2f}%")
    print(f"Lower bound: {lower_bound:.2f}")
    print(f"Upper bound: {upper_bound:.2f}")
    
    print("\nOutlier Summary Statistics (Before Replacement):")
    print(original_outliers['systemMarginalPrice'].describe())
    
    print("\nReplacement Summary Statistics:")
    print(data_cleaned.loc[original_outliers.index, 'systemMarginalPrice'].describe())

    return data_cleaned

def main():
    # Fetch and merge EPIAŞ data
    merged_epias_data = get_merged_epias_data(start_date, end_date)
    
    # Detect outliers in systemMarginalPrice using IQR method
    Q1 = merged_epias_data['systemMarginalPrice'].quantile(0.25)
    Q3 = merged_epias_data['systemMarginalPrice'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR 
    upper_bound = Q3 + 1.5 * IQR

    
    def is_outlier(value):
        return value < lower_bound or value > upper_bound

    # Identify outliers
    outliers_mask = (
        (merged_epias_data['systemMarginalPrice'] < lower_bound) | 
        (merged_epias_data['systemMarginalPrice'] > upper_bound)
    )
    outliers = merged_epias_data[outliers_mask]
    print(len(outliers))
  
    path = os.getenv('DATA_PATH')
    #data_cleaned.to_csv(path + "/data/processed/cleaned_epias_data.csv", index=True)
    print("\nCleaned EPIAŞ data saved to CSV.")

if __name__ == "__main__":
    main()