import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller


# Load the data
df = pd.read_csv(r'C:\Users\Salih\Documents\GitHub\prediction-system-marginal-price\data\raw\combined_data.csv')
print("Dataset Shape:", df.shape)
print(df.info())

columns_to_drop = ['izmir_snow', 'istanbul_tsun', 'ankara_tsun', 'bursa_tsun',
                 'antalya_tsun', 'izmir_wpgt', 'izmir_tsun', 'istanbul_snow',
                 'ankara_snow', 'antalya_snow', 'bursa_snow', 'bursa_wpgt',
                 'istanbul_wpgt','Volume_x','Volume_y']

# Kolonları silme
df = df.drop(columns=columns_to_drop)
print("Yeni veri seti boyutu:", df.shape)

plt.figure(figsize=(15, 6))
plt.plot(df['systemMarginalPrice'])
plt.title('System Marginal Price Over Time')
plt.xlabel('Time')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()