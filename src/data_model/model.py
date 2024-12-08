import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from abc import ABC, abstractmethod

class BaseTimeSeriesModel(ABC):
    """
    Abstract base class for time series forecasting models.
    """
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def evaluate(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        print('y_true mean:', np.mean(y_true))
        print('y_pred mean:', np.mean(y_pred))
        mae_percentage = mae / np.mean(y_true) * 100
        mape = mean_absolute_percentage_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mase = mae / np.mean(np.abs(y_true - np.roll(y_true, 1)))
        r2 = r2_score(y_true, y_pred)
        metrics = {'MAE': mae, 'MAE Percentage': mae_percentage, 'MAPE': mape, 
                   'RMSE': rmse, 'MASE': mase, 'R2': r2}
        print(metrics)

    def plot_results(self, y_true, y_pred):
        """
        Plots the actual vs predicted values as a line chart.
        
        Args:
            y_true (np.ndarray or pd.Series): Actual target values.
            y_pred (np.ndarray or pd.Series): Predicted target values.
        """
        prediction = pd.DataFrame({'y-pred':y_pred}, index=y_true.index)

        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='Actual', color='blue')
        plt.plot(prediction['y-pred'], label='Predicted', color='orange')
        plt.title('Actual vs Predicted Time Series')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()

class ARIMAModel(BaseTimeSeriesModel):
    def __init__(self, order=(5,1,0)):
        self.order = order
        self.model = None
        self.results = None

    def fit(self, y):
        self.model = ARIMA(y, order=self.order)
        self.results = self.model.fit()
        print("ARIMA model fitted.")

    def predict(self, steps):
        forecast = self.results.forecast(steps=steps)
        return forecast

class SARIMAModel(BaseTimeSeriesModel):
    def __init__(self, order=(1,1,1), seasonal_order=(1,1,1,12)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.results = None

    def fit(self, y):
        self.model = SARIMAX(y, order=self.order, seasonal_order=self.seasonal_order)
        self.results = self.model.fit()
        print("SARIMA model fitted.")

    def predict(self, steps):
        forecast = self.results.forecast(steps=steps)
        return forecast

class ExponentialSmoothingModel(BaseTimeSeriesModel):
    def __init__(self, trend='add', seasonal='add', seasonal_periods=12):
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.model = None
        self.results = None

    def fit(self, y):
        self.model = ExponentialSmoothing(y, trend=self.trend, seasonal=self.seasonal, seasonal_periods=self.seasonal_periods)
        self.results = self.model.fit()
        print("Exponential Smoothing model fitted.")

    def predict(self, steps):
        forecast = self.results.forecast(steps=steps)
        return forecast

class RandomForestModel(BaseTimeSeriesModel):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2):
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)

    def fit(self, X, y):
        self.model.fit(X, y)
        print("Random Forest model fitted.")

    def predict(self, X):
        return self.model.predict(X)

class GradientBoostingModel(BaseTimeSeriesModel):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)

    def fit(self, X, y):
        self.model.fit(X, y)
        print("Gradient Boosting model fitted.")

    def predict(self, X):
        return self.model.predict(X)

class XGBoostModel(BaseTimeSeriesModel):
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1):
        self.model = XGBRegressor(objective='reg:squarederror', n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)

    def fit(self, X, y):
        self.model.fit(X, y)
        print("XGBoost model fitted.")

    def predict(self, X):
        return self.model.predict(X)

class LSTMModel:
    def __init__(self, target, seq_length=24):
        self.target = target
        self.seq_length = seq_length
        self.scaler = MinMaxScaler()
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.history = None

    def prepare_data(self, X_train, y_train, X_test, y_test):
        """
        Prepares and scales the training and testing data.
        
        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            X_test (pd.DataFrame): Testing features.
            y_test (pd.Series): Testing target.
        """
        # Forward fill to handle missing values
        X_train.ffill(inplace=True)
        X_test.ffill(inplace=True)
        
        # Scale the data
        self.scaler.fit(X_train)
        scaled_X_train = self.scaler.transform(X_train)
        scaled_X_test = self.scaler.transform(X_test)
        
        # Create sequences
        self.X_train, self.y_train = self.create_sequences(scaled_X_train, y_train)
        self.X_test, self.y_test = self.create_sequences(scaled_X_test, y_test)
        print("LSTM data prepared.")

    @staticmethod
    def create_sequences(X, y, seq_length=24):
        """
        Creates sequences of data for LSTM input.
        
        Args:
            X (np.ndarray): Feature data.
            y (np.ndarray): Target data.
            seq_length (int): Length of the sequences.
            
        Returns:
            tuple: Arrays of input sequences and corresponding targets.
        """
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i+seq_length])
            y_seq.append(y[i+seq_length])
        return np.array(X_seq), np.array(y_seq)

    def build_model(self):
        """
        Builds the LSTM model architecture.
        """
        self.model = Sequential()
        self.model.add(Input(shape=(self.X_train.shape[1], self.X_train.shape[2])))
        self.model.add(LSTM(100, activation='relu', return_sequences=True))
        self.model.add(LSTM(50, activation='relu'))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')
        print("LSTM model built.")
    
    def train_model(self, epochs=100, batch_size=64):
        """
        Trains the LSTM model with early stopping.
        
        Args:
            epochs (int): Number of training epochs.
            batch_size (int): Size of the training batches.
        """
        early_stop = EarlyStopping(monitor='val_loss', patience=10)
        self.history = self.model.fit(
            self.X_train, self.y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_data=(self.X_test, self.y_test),
            callbacks=[early_stop]
        )
        print("LSTM model trained.")
    
    def predict(self, X):
        """
        Generates predictions using the trained LSTM model.
        
        Args:
            X (np.ndarray): Input sequences for prediction.
            
        Returns:
            np.ndarray: Predicted values.
        """
        scaled_X = self.scaler.transform(X)
        X_seq, _ = self.create_sequences(scaled_X, np.zeros(len(scaled_X)))
        return self.model.predict(X_seq)
    
    def plot_history(self):
        """
        Plots the training and validation loss history.
        """
        plt.figure(figsize=(10,6))
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.title('LSTM Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
    
    def evaluate(self, y_true, y_pred):
        """
        Evaluates the model's performance using various metrics.
        
        Args:
            y_true (np.ndarray): Actual target values.
            y_pred (np.ndarray): Predicted target values.
            
        Returns:
            dict: Dictionary containing MAE, MAPE, RMSE, MASE, and R2 scores.
        """
        mae = mean_absolute_error(y_true, y_pred)
        mape = mae / np.mean(y_true) * 100
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mase = mae / np.mean(np.abs(y_true - np.roll(y_true, 1)))
        r2 = r2_score(y_true, y_pred)
        
        return {
            'MAE': mae,
            'MAPE': mape,
            'RMSE': rmse,
            'MASE': mase,
            'R2': r2
        }

class EnsembleModel(BaseTimeSeriesModel):
    def __init__(self, models):
        """
        models: list of BaseTimeSeriesModel instances
        """
        self.models = models

    def predict(self, X):
        predictions = []
        for model in self.models:
            preds = model.predict(X)
            predictions.append(preds)
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred

    def evaluate(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mae_percentage = mae / np.mean(y_true) * 100
        mape = mean_absolute_percentage_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mase = mae / np.mean(np.abs(y_true - np.roll(y_true, 1)))
        r2 = r2_score(y_true, y_pred)
        return {'MAE': mae, 'MAE Percentage': mae_percentage, 'MAPE': mape, 'RMSE': rmse, 'MASE': mase, 'R2': r2}