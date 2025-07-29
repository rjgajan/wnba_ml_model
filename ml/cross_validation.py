import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import numpy as np

def cross_validate_model(df):
    # features / target
    features = ['rest_diff', 'OPP_PACE', 'avg_prev_5', 
                'avg_prev_15', 'HOME_AWAY', 'OPP_DEF_RATING', 'team_rest_days']
    target = 'PTS'
    df = df.dropna(subset=features + [target])
    X = df[features]
    y = df[target]

    # preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('home_away_enc', OneHotEncoder(handle_unknown='ignore'), ['HOME_AWAY'])
        ],
        remainder='passthrough'
    )

    # XGBRegressor pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(
            n_estimators=530,
            learning_rate=0.01001361825891669,
            max_depth=3,
            subsample=0.7514892897979619,
            colsample_bytree=0.7348989896501112,
            gamma=1.99332691197874,
            min_child_weight=2,
            reg_lambda=2.865331577453844,
            reg_alpha=5.275622148785655,
            random_state=42
        ))
    ])

    # 5-fold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error')

    # convert negative scores to positive RMSE
    rmse_scores = -scores
    print(f"Cross-Validated RMSE scores: {rmse_scores}")
    print(f"Mean RMSE: {rmse_scores.mean():.3f}, Std Dev: {rmse_scores.std():.3f}")

    return rmse_scores.mean()

# main loop
if __name__ == "__main__":
    df = pd.read_csv('processed_2024_2025.csv')  # Load data
    avg_rmse = cross_validate_model(df)
    print(f"Average RMSE across folds: {avg_rmse:.2f}")