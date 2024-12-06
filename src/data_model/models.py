import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import plotly.express as px
import plotly.graph_objects as go   
from plotly.subplots import make_subplots
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from keras.callbacks import EarlyStopping

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.filepath)
        self._process_datetime()
        self._set_index()
        self._drop_columns_initial()
        self._handle_missing_values()
        print("Data loaded successfully.")
        return self.df

    def _process_datetime(self):
        self.df['datetime'] = pd.to_datetime(self.df['Unnamed: 0'], format='%Y-%m-%d %H:%M:%S')
        self.df['week'] = self.df['datetime'].dt.isocalendar().week
        self.df['year'] = self.df['datetime'].dt.year
        self.df['day'] = self.df['datetime'].dt.day
        self.df['month'] = self.df['datetime'].dt.month
        self.df['is_weekend'] = self.df['datetime'].dt.weekday.isin([5, 6])

    def _set_index(self):
        self.df.set_index('datetime', inplace=True)
        # Explicitly set frequency to hourly to eliminate warnings
        self.df = self.df.asfreq('H')
        # Handle any potential missing timestamps after setting frequency
        self.df.ffill(inplace=True)
    
    def _drop_columns_initial(self):
        self.df.drop(['Unnamed: 0'], axis=1, inplace=True)

    def _handle_missing_values(self):
        self.df['izmir_coco'].fillna(self.df['izmir_coco'].mean(), inplace=True)
        self.df['izmir_pres'].fillna(self.df['izmir_pres'].mean(), inplace=True)
        print("Missing values handled.")

class FeatureEngineer:
    def __init__(self, df):
        self.df = df

    def add_direction_dummies(self):
        direction_dummies = pd.get_dummies(self.df['systemDirection'], prefix='direction', dtype=float)
        self.df = pd.concat([self.df, direction_dummies], axis=1).drop(columns=['systemDirection']).copy()

    def drop_unnecessary_columns(self):
        columns_to_drop = [
            'izmir_snow', 'istanbul_tsun', 'ankara_tsun', 'bursa_tsun', 
            'antalya_tsun', 'izmir_wpgt', 'izmir_tsun', 'istanbul_snow',
            'ankara_snow', 'antalya_snow', 'bursa_snow', 'bursa_wpgt',
            'istanbul_wpgt','Volume','smpDirectionId'
        ]
        self.df = self.df.drop(columns=columns_to_drop)
        print("Unnecessary columns dropped.")

    def calculate_average_prices(self):
        self.df['mean_ghi'] = self.df[['istanbul_ghi','ankara_ghi','bursa_ghi','izmir_ghi','antalya_ghi']].mean(axis=1)
        self.df['mean_dni'] = self.df[['istanbul_dni','ankara_dni','bursa_dni','izmir_dni','antalya_dni']].mean(axis=1)
        self.df['mean_dwpt'] = self.df[['ankara_dwpt','istanbul_dwpt','bursa_dwpt','izmir_dwpt','antalya_dwpt']].mean(axis=1)
        print("Average prices calculated.")

    def add_hourly_lags(self, columns, lag_hours=24):
        for col in columns:
            self.df[f'{col}_lag_{lag_hours}h'] = self.df[col].shift(lag_hours)
        self.df.drop(columns=columns, inplace=True)
        self.df = self.df.iloc[lag_hours:]
        print(f"Hourly lags added for columns: {columns}")

class CorrelationAnalyzer:
    def __init__(self, df, target):
        self.df = df
        self.target = target
        self.correlation_df = None

    def compute_correlations(self):
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        correlations_pearson = self.df[numeric_columns].corr(method='pearson')[self.target]
        correlations_kendall = self.df[numeric_columns].corr(method='kendall')[self.target]
        correlations_spearman = self.df[numeric_columns].corr(method='spearman')[self.target]
        
        self.correlation_df = pd.concat(
            [correlations_pearson, correlations_kendall, correlations_spearman], 
            axis=1, 
            keys=['Pearson', 'Kendall', 'Spearman']
        )
        
        self.correlation_df = self.correlation_df.abs().sort_values(by='Kendall', ascending=False)
        top_corr_df = self.correlation_df[
            (self.correlation_df['Pearson'] > 0.1) | 
            (self.correlation_df['Kendall'] > 0.1) | 
            (self.correlation_df['Spearman'] > 0.1)
        ]
        top_corr_df_20 = top_corr_df.head(20)
        return top_corr_df_20

    def plot_correlations(self, top_corr_df_20):
        plt.figure(figsize=(14, 10))
        top_corr_df_20 = top_corr_df_20.reset_index().melt(id_vars="index", var_name="Correlation_Type", value_name="Correlation_Value")
        
        sns.barplot(x="Correlation_Value", y="index", hue="Correlation_Type", data=top_corr_df_20)
        plt.title('Top Correlations with System Marginal Price (Grouped by Correlation Type)')
        plt.xlabel('Correlation Value')
        plt.ylabel('Features')
        plt.legend(title='Correlation Type')
        plt.tight_layout()
        plt.show()
        print("Correlation plot displayed.")

class ModelTrainer:
    def __init__(self, df, target):
        self.df = df
        self.target = target

    def train_arima(self, order=(5, 1, 0), steps=24):
        train_size = int(len(self.df) * 0.8)
        train, test = self.df[self.target][:train_size], self.df[self.target][train_size:]
        
        # Explicitly set frequency in ARIMA
        model = ARIMA(train, order=order, freq='H')
        model_fit = model.fit()
        
        forecast = model_fit.forecast(steps=steps)
        actual = test[:steps]
        mae = mean_absolute_error(actual, forecast)
        mape = self.mean_absolute_percentage_error(actual, forecast)
        
        self._plot_forecast(actual, forecast)
        
        print(f"24 Saatlik Tahmin: {forecast}")
        print(f"Gerçek Değerler: {actual.values}")
        print(f"Ortalama Mutlak Hata (MAE): {mae}")
        print(f"Ortalama Yüzde Hata (MAPE): {mape:.2f}%")
        return forecast

    def _plot_forecast(self, actual, forecast):
        plt.figure(figsize=(10, 5))
        plt.plot(actual.index, actual, label='Gerçek')
        plt.plot(actual.index, forecast, label='Tahmin', linestyle='--')
        plt.title('24 Saatlik Tahmin')
        plt.xlabel('Zaman')
        plt.ylabel(self.target)
        plt.legend()
        plt.show()
        print("Forecast plot displayed.")

    def train_regression_models(self):
        features = self.df.drop(columns=[self.target])
        target = self.df[self.target]
        
        # Create lag features
        for lag in range(1, 25):
            features[f'{self.target}_lag_{lag}'] = target.shift(lag)
        
        features.dropna(inplace=True)
        target = target[features.index]
        
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)
        
        # Feature Scaling
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models with hyperparameter grids
        models = {
            'XGBoost': {
                'model': XGBRegressor(objective='reg:squarederror', random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            },
            'RandomForest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                }
            }
        }
        
        results = {}
        for name, config in models.items():
            print(f"Training {name}...")
            grid = GridSearchCV(config['model'], config['params'], cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
            grid.fit(X_train_scaled, y_train)
            best_model = grid.best_estimator_
            preds = best_model.predict(X_test_scaled)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mape = self.mean_absolute_percentage_error(y_test, preds)
            results[name] = {'Best Params': grid.best_params_, 'RMSE': rmse, 'MAPE': mape}
            print(f'{name} - Best Params: {grid.best_params_}, RMSE: {rmse}, MAPE: {mape}')
        
        return results

    @staticmethod
    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        non_zero_indices = y_true != 0
        return np.mean(np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])) * 100

class LSTMModel:
    def __init__(self, df, target, seq_length=24):
        self.df = df
        self.target = target
        self.seq_length = seq_length
        self.scaler = MinMaxScaler()
        self.model = None

    def prepare_data(self):
        self.df.ffill(inplace=True)
        scaled_data = self.scaler.fit_transform(self.df)
        
        X, y = self.create_sequences(scaled_data, self.seq_length, self.df.columns.get_loc(self.target))
        
        train_size = int(len(X) * 0.8)
        self.X_train, self.X_test = X[:train_size], X[train_size:]
        self.y_train, self.y_test = y[:train_size], y[train_size:]
        print("LSTM data prepared.")

    @staticmethod
    def create_sequences(data, seq_length, target_index):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length, target_index])
        return np.array(X), np.array(y)

    def build_model(self):
        self.model = Sequential()
        self.model.add(Input(shape=(self.X_train.shape[1], self.X_train.shape[2])))
        self.model.add(LSTM(100, activation='relu', return_sequences=True))
        self.model.add(LSTM(50, activation='relu'))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')
        print("LSTM model built.")
    
    def train_model(self, epochs=100, batch_size=64):
        early_stop = EarlyStopping(monitor='val_loss', patience=10)
        self.model.fit(
            self.X_train, self.y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_data=(self.X_test, self.y_test),
            callbacks=[early_stop]
        )
        print("LSTM model trained.")

def main():
    # Initialize DataLoader
    loader = DataLoader(r'C:\Users\Salih\Documents\GitHub\prediction-system-marginal-price\data\raw\combined_data.csv')
    df = loader.load_data()

    # Feature Engineering
    engineer = FeatureEngineer(df)
    engineer.add_direction_dummies()
    engineer.drop_unnecessary_columns()
    engineer.calculate_average_prices()
    columns_to_lag = [
        'upRegulationNet', 'downRegulationNet', 'upRegulationZeroCoded',
        'upRegulationDelivered','downRegulationZeroCoded','downRegulationDelivered',
        'direction_Dengede', 'direction_Enerji Açığı','direction_Enerji Fazlası'
    ]
    engineer.add_hourly_lags(columns_to_lag, lag_hours=24)
    df_with_lags = engineer.df

    # Correlation Analysis
    analyzer = CorrelationAnalyzer(df_with_lags, target='systemMarginalPrice')
    top_corr_df_20 = analyzer.compute_correlations()
    analyzer.plot_correlations(top_corr_df_20)

    # Model Training
    trainer = ModelTrainer(df_with_lags, target='systemMarginalPrice')
    trainer.train_arima()
    trainer.train_regression_models()

    # LSTM Modeling
    lstm = LSTMModel(df_with_lags, target='systemMarginalPrice')
    lstm.prepare_data()
    lstm.build_model()
    lstm.train_model()

if __name__ == "__main__":
    main()