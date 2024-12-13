import numpy as np
from src.data_model.models import DataLoader, FeatureEngineer, CorrelationAnalyzer, ModelTrainer
from src.data_model.model import XGBoostModel, RandomForestModel, GradientBoostingModel
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error, make_scorer, mean_absolute_error

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
    #engineer.add_hourly_lags(columns_to_lag, lag_hours=24)
    df_with_lags = engineer.df
    X = df_with_lags.drop(columns=['systemMarginalPrice'])
    y = df_with_lags['systemMarginalPrice']

    # Split data
    # For time series, it's better to keep the temporal order intact
    split_point = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

    # Define parameter grids
    param_grid_xgb = {
        'n_estimators': [100, 200, 300, 400, 500,600,700],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
    }

    param_grid_rf = {
        'bootstrap': [True, False],
        'max_depth': [3, 5, 7, 9,10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        'max_features': ['auto', 'sqrt'],
        'n_estimators': [200, 400, 600, 800, 1000]    
    }

    param_grid_gb = {
        'n_estimators': [100, 200, 300, 400, 500,600,700],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
    }

    # Initialize models
    xgb = XGBoostModel()
    rf = RandomForestModel()
    gb = GradientBoostingModel()

    # Define TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)

    #mape scorer
    mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)

    # Initialize GridSearchCV for each model with TimeSeriesSplit and MAPE
    grid_xgb = GridSearchCV(estimator=xgb.model, param_grid=param_grid_xgb, 
                            cv=tscv, scoring=mape_scorer, n_jobs=-1,verbose=10)
    grid_rf = GridSearchCV(estimator=rf.model, param_grid=param_grid_rf, 
                           cv=tscv, scoring=mape_scorer, n_jobs=-1,verbose=10)
    grid_gb = GridSearchCV(estimator=gb.model, param_grid=param_grid_gb, 
                           cv=tscv, scoring=mape_scorer, n_jobs=-1,verbose=10)

    # Fit GridSearchCV
    grid_xgb.fit(X_train, y_train)
    best_xgb = grid_xgb.best_estimator_
    xgb_preds = best_xgb.predict(X_test)
    mae = mean_absolute_error(y_test, xgb_preds)
    xgb_mae_percentage = mae / np.mean(y_test) * 100  
    xgb_mape = mean_absolute_percentage_error(y_test, xgb_preds)
    print('XGBoost Best Params:', grid_xgb.best_params_)
    print('XGBoost MAPE:', xgb_mape)
    print('XGBoost MAE Percentage:', xgb_mae_percentage)

    grid_rf.fit(X_train, y_train)
    best_rf = grid_rf.best_estimator_
    rf_preds = best_rf.predict(X_test)
    mae = mean_absolute_error(y_test, rf_preds)
    rf_mae_percentage = mae / np.mean(y_test) * 100  
    rf_mape = mean_absolute_percentage_error(y_test, rf_preds)
    print('Random Forest Best Params:', grid_rf.best_params_)
    print('Random Forest MAPE:', rf_mape)   
    print('Random Forest MAE Percentage:', rf_mae_percentage)
    grid_gb.fit(X_train, y_train)
    best_gb = grid_gb.best_estimator_
    gb_preds = best_gb.predict(X_test)
    mae = mean_absolute_error(y_test, gb_preds)
    gb_mae_percentage = mae / np.mean(y_test) * 100  
    gb_mape = mean_absolute_percentage_error(y_test, gb_preds)


    print('XGBoost Best Params:', grid_xgb.best_params_)
    print('XGBoost MAPE:', xgb_mape)
    print('XGBoost MAE Percentage:', xgb_mae_percentage)

    print('Random Forest Best Params:', grid_rf.best_params_)
    print('Random Forest MAPE:', rf_mape)
    print('Random Forest MAE Percentage:', rf_mae_percentage)

    print('Gradient Boosting Best Params:', grid_gb.best_params_)
    print('Gradient Boosting MAPE:', gb_mape)
    print('Gradient Boosting MAE Percentage:', gb_mae_percentage)



if __name__ == "__main__":
    main()
