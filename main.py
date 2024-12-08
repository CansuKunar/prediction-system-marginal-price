from src.data_model.models import (DataLoader, FeatureEngineer, CorrelationAnalyzer, ModelTrainer)
from src.data_model.model import (ARIMAModel, XGBoostModel, LSTMModel, EnsembleModel,SARIMAModel,RandomForestModel,GradientBoostingModel)
from sklearn.model_selection import train_test_split
import numpy as np

def main():
    loader = DataLoader(r'C:\Users\Salih\Documents\GitHub\prediction-system-marginal-price\data\raw\combined_data.csv')
    df = loader.load_data()

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
    #analyzer.plot_correlations(top_corr_df_20)

    # Prepare features and target
    X = df_with_lags.drop(columns=['systemMarginalPrice'])
    y = df_with_lags['systemMarginalPrice']

    # Split data
    split_point = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

    # Initialize models
    arima = ARIMAModel(order=(5,1,0))
    xgb = XGBoostModel(n_estimators=200, max_depth=5, learning_rate=0.1)
    sarima = SARIMAModel(order=(1,1,1), seasonal_order=(1,1,1,12))
    rf = RandomForestModel(n_estimators=100, max_depth=10)
    gb = GradientBoostingModel(n_estimators=100, learning_rate=0.1, max_depth=5)

    # Fit models
    X_train_lstm = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    arima.fit(y_train)
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    xgb.plot_results(y_test, xgb_preds)
    sarima.fit(y_train)
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)               

    # Make predictions
    arima_preds = arima.predict(steps=len(X_test))
    xgb_preds = xgb.predict(X_test)
    sarima_preds = sarima.predict(steps=len(X_test))
    rf_preds = rf.predict(X_test)
    gb_preds = gb.predict(X_test)

    # Evaluate models
    print('ARIMA Evaluation:',arima.evaluate(y_test, arima_preds))
    print('XGBoost Evaluation:',xgb.evaluate(y_test, xgb_preds))
    print('SARIMA Evaluation:',sarima.evaluate(y_test, sarima_preds))
    print('Random Forest Evaluation:',rf.evaluate(y_test, rf_preds))
    print('Gradient Boosting Evaluation:',gb.evaluate(y_test, gb_preds))

    arima.plot_results(y_test, arima_preds)
    xgb.plot_results(y_test, xgb_preds)
    sarima.plot_results(y_test, sarima_preds)
    rf.plot_results(y_test, rf_preds)
    gb.plot_results(y_test, gb_preds)

    # Ensemble Model
    predictions = [xgb_preds, rf_preds, gb_preds]
    ensemble_pred = np.mean(predictions, axis=0)
    print('Ensemble Evaluation:',gb.evaluate(y_test, ensemble_pred))



if __name__ == "__main__":
    main()  