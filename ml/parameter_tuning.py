import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor
import optuna

def preprocess_and_tune(df):
    # Features and target
    features = ['rest_diff', 'OPP_PACE', 'avg_prev_5', 
                'avg_prev_15', 'HOME_AWAY', 'OPP_DEF_RATING', 'team_rest_days']
    target = 'PTS'

    # Drop rows with missing values in features or target
    df = df.dropna(subset=features + [target])
    X = df[features]
    y = df[target]

    # Train/test split (random for now, can do time-based if needed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # One-hot encode HOME_AWAY
    preprocessor = ColumnTransformer(
        transformers=[
            ('home_away_enc', OneHotEncoder(handle_unknown='ignore'), ['HOME_AWAY'])
        ],
        remainder='passthrough'
    )

    def objective(trial):
        # Hyperparameter search space
        params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True)
        }


        # Pipeline with XGBoost regressor
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', XGBRegressor(objective='reg:squarederror', eval_metric='rmse', **params))
        ])

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, preds)
        return rmse

    # Run Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=300)  # Increase n_trials for better results

    print("Best hyperparameters:", study.best_trial.params)

    # Train final model with best parameters
    best_params = study.best_trial.params
    final_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(objective='reg:squarederror', eval_metric='rmse', **best_params))
    ])
    final_model.fit(X_train, y_train)

    preds = final_model.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)
    print(f"Final Test RMSE: {rmse:.2f}")

    return final_model

if __name__ == "__main__":
    df = pd.read_csv('processed_2024_2025.csv')
    model = preprocess_and_tune(df)

    # Example prediction
    example_input = pd.DataFrame([{
        'OPP_DEF_RATING': 99.4,
        'avg_prev_15': 22.27,
        'avg_prev_5': 24.2,
        'HOME_AWAY': 'Away',
        'rest_diff': 0,
        'OPP_PACE': 80,
        'team_rest_days': 5
    }])
    prediction = model.predict(example_input)
    print(f"Predicted next game points: {prediction[0]:.2f}")