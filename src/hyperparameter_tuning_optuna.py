import numpy as np
import optuna
from optuna.samplers import TPESampler
from data_model.models import DataLoader, FeatureEngineer, CorrelationAnalyzer, ModelTrainer
from data_model.model import XGBoostModel, RandomForestModel, GradientBoostingModel
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import lightgbm as lgb

def load_and_prepare_data():
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
    X = df_with_lags.drop(columns=['systemMarginalPrice'])
    y = df_with_lags['systemMarginalPrice']

    split_point = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mae_percentage = mae / np.mean(y_test) * 100
    mape = mean_absolute_percentage_error(y_test, preds)
    return mae, mae_percentage, mape

def optimize_xgboost(trial, X_train, y_train, tscv):
    param = {
        "n_estimators": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
    }
    model = XGBoostModel(**param)
    mae_scores = []
    for train_idx, valid_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        mae_scores.append(mae)
    return np.mean(mae_scores)

def optimize_random_forest(trial, X_train, y_train, tscv):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 2000, step=200),
        'max_depth': trial.suggest_int('max_depth', 3, 100),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
    }
    model = RandomForestModel(**param)
    mae_scores = []
    for train_idx, valid_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        mae_scores.append(mae)
    return np.mean(mae_scores)

def optimize_gradient_boosting(trial, X_train, y_train, tscv):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0)    }
    model = GradientBoostingModel(**param)
    mae_scores = []
    for train_idx, valid_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        mae_scores.append(mae)
    return np.mean(mae_scores)

def main():
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    tscv = TimeSeriesSplit(n_splits=5)

    # Optimize XGBoost
    study_xgb = optuna.create_study(direction='minimize', sampler=TPESampler())
    study_xgb.optimize(lambda trial: optimize_xgboost(trial, X_train, y_train, tscv), n_trials=50)
    best_params_xgb = study_xgb.best_params
    print('XGBoost Best Params:', best_params_xgb)

    # Train best XGBoost model
    best_xgb = XGBoostModel(**best_params_xgb)
    best_xgb.fit(X_train, y_train)
    xgb_mae, xgb_mae_percentage, xgb_mape = evaluate_model(best_xgb, X_test, y_test)
    print('XGBoost MAPE:', xgb_mape)
    print('XGBoost MAE Percentage:', xgb_mae_percentage)

    # Optimize Random Forest
    study_rf = optuna.create_study(direction='minimize', sampler=TPESampler())
    study_rf.optimize(lambda trial: optimize_random_forest(trial, X_train, y_train, tscv), n_trials=50)
    best_params_rf = study_rf.best_params
    print('Random Forest Best Params:', best_params_rf)

    # Train best Random Forest model
    best_rf = RandomForestModel(**best_params_rf)
    best_rf.fit(X_train, y_train)
    rf_mae, rf_mae_percentage, rf_mape = evaluate_model(best_rf, X_test, y_test)
    print('Random Forest MAPE:', rf_mape)
    print('Random Forest MAE Percentage:', rf_mae_percentage)

    # Optimize Gradient Boosting
    study_gb = optuna.create_study(direction='minimize', sampler=TPESampler())
    study_gb.optimize(lambda trial: optimize_gradient_boosting(trial, X_train, y_train, tscv), n_trials=50)
    best_params_gb = study_gb.best_params
    print('Gradient Boosting Best Params:', best_params_gb)

    # Train best Gradient Boosting model
    best_gb = GradientBoostingModel(**best_params_gb)
    best_gb.fit(X_train, y_train)
    gb_mae, gb_mae_percentage, gb_mape = evaluate_model(best_gb, X_test, y_test)
    print('Gradient Boosting MAPE:', gb_mape)
    print('Gradient Boosting MAE Percentage:', gb_mae_percentage)

if __name__ == "__main__":
    main() 